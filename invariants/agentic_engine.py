import torch
import torch.nn.functional as F
from invariants.engine import _inputs, _generate_ids
from invariants.social_hunt import get_steer_vector
from invariants.multi_domain_benchmark import DOMAINS
from invariants.cognitive_cache import CognitiveCache

# Global cache instance
_global_cache = CognitiveCache()

def _entropy_from_logits(logits):
    probs = F.softmax(logits.float(), dim=-1)
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)

def _add_last_token_delta(h, delta):
    d = delta.to(h.device)
    if d.ndim == 1:
        d = d.view(1, 1, -1)
    elif d.ndim == 2:
        d = d.unsqueeze(1)
    elif d.ndim == 3 and d.shape[1] != 1:
        d = d[:, -1:, :]

    d = d.to(h.dtype)
    add = torch.zeros_like(h)
    add[:, -1:, :] = d
    return h + add

def get_agentic_handles(
    M,
    vecs,
    belief_vec=None,
    humility_vec=None,
    alpha=0.5,
    epsilon=0.05,
    entropy_threshold=2.0,
    max_loops=3,
    force_synthesis=False,
    cache_enabled=True,
    cache_write_enabled=False,
    cache_verified_only=True,
    synthesis_enabled=True,
    max_synthesis_events=1,
    synthesis_recorder=None,
    max_routing_events=4,
):
    """
    Parallel Latent Search (ToT)
    Dynamically clones the hidden state into 3 branches at the plateau.
    Injects 3 different optimizers. Evaluates entropy. Keeps the best.
    """
    handles = []
    
    state = {
        "prev_h": None,
        "start_layer": -1,
        "end_layer": -1,
        "loop_count": 0,
        "total_loops_this_token": 0,
        "branch_names": list(vecs.keys()),
        "belief_vec": belief_vec,
        "humility_vec": humility_vec,
        "synthesis_events": 0,
        "routing_events": 0,
    }
    
    def make_hook(l_idx):
        def hook(module, args, kwargs, out):
            if isinstance(out, tuple):
                h = out[0]
            else:
                h = out
                
            if l_idx == 0:
                state["prev_h"] = h.detach().clone()
                state["start_layer"] = -1
                state["end_layer"] = -1
                state["total_loops_this_token"] = 0
                return out
                
            prev_h = state["prev_h"]
            
            curr_token_h = h.float()
            prev_token_h = prev_h.float()
            
            cos_sim = F.cosine_similarity(curr_token_h, prev_token_h, dim=-1)
            velocity = 1.0 - cos_sim
            last_token_velocity = velocity[:, -1].mean().item()
            
            if (last_token_velocity < epsilon or (force_synthesis and l_idx == 1)) and state["start_layer"] == -1:
                state["start_layer"] = l_idx
                
            # Trigger if velocity spikes, OR if we reach the final layer of the network
            at_boundary = (last_token_velocity > epsilon) or (l_idx == len(M.model.model.layers) - 1)
            
            if state["start_layer"] != -1 and at_boundary and state["end_layer"] == -1:
                state["end_layer"] = l_idx
                
                # We have reached the post-hoc boundary! 
                # Let's project the current baseline state to check entropy
                h_last = h[:, -1:, :]
                h_norm = M.model.model.norm(h_last.to(M.model.dtype))
                logits = M.model.lm_head(h_norm)
                entropy = _entropy_from_logits(logits).mean().item()
                
                if (
                    entropy > entropy_threshold
                    and state["total_loops_this_token"] < max_loops
                    and state["routing_events"] < max_routing_events
                ):
                    state["total_loops_this_token"] += 1
                    state["routing_events"] += 1
                    
                    # 1. Parallel Branching!
                    # h is shape [batch, seq, dim]. Usually batch=1.
                    # We repeat it to [3, seq, dim]
                    h_parallel = h.repeat(3, 1, 1)
                    
                    # 2. Inject vectors at the start of the plateau
                    # (To simulate injecting at the start, we just add the vectors to the hidden state now, 
                    # before routing it back through the layers)
                    v_soc = vecs["Social"].to(h.device).to(h.dtype)
                    v_cre = vecs["Creative"].to(h.device).to(h.dtype)
                    v_ana = vecs["Analytical"].to(h.device).to(h.dtype)
                    
                    # Add to the last token of each branch
                    h_parallel[0, -1, :] += alpha * (v_soc / v_soc.norm())
                    h_parallel[1, -1, :] += alpha * (v_cre / v_cre.norm())
                    h_parallel[2, -1, :] += alpha * (v_ana / v_ana.norm())
                    
                    # 3. Parallel Recurrent Loop
                    for i in range(state["start_layer"], l_idx + 1):
                        layer_kwargs = dict(kwargs)
                        layer_kwargs["use_cache"] = False
                        for cache_key in ("past_key_value", "past_key_values"):
                            if cache_key in layer_kwargs:
                                del layer_kwargs[cache_key]
                        for key in ("attention_mask", "position_ids"):
                            value = layer_kwargs.get(key)
                            if torch.is_tensor(value) and value.shape[0] == 1 and h_parallel.shape[0] != 1:
                                layer_kwargs[key] = value.repeat(h_parallel.shape[0], *([1] * (value.ndim - 1)))
                        
                        layer_out = M.model.model.layers[i](h_parallel, *args[1:], **layer_kwargs)
                        h_parallel = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                    
                    # 4. Evaluate Entropy of Branches
                    h_parallel_last = h_parallel[:, -1:, :]
                    h_norm_parallel = M.model.model.norm(h_parallel_last.to(M.model.dtype))
                    logits_parallel = M.model.lm_head(h_norm_parallel)
                    entropies = _entropy_from_logits(logits_parallel).squeeze(1) # [3]
                    
                    # 5. Selection
                    best_idx = torch.argmin(entropies).item()
                    best_entropy = entropies[best_idx].item()
                    
                    print(f"    [Agentic ToT] Token Loop {state['total_loops_this_token']} | "
                          f"Soc: {entropies[0]:.2f}, Cre: {entropies[1]:.2f}, Ana: {entropies[2]:.2f} "
                          f"-> WINNER: {state['branch_names'][best_idx]} (Entropy: {best_entropy:.2f})")
                    
                    # Collapse back to batch=1 with the winning state
                    routed_h = h_parallel[best_idx].unsqueeze(0)
                    
                    # USER REQUEST: "If the model isn't satisfied, it shouldn't be confident in its answer"
                    # UPDATE: "but this learning has to be influenced by the model's internal skepticism."
                    # Dynamic Test-Time Layer Synthesis
                    should_synthesize = (
                        synthesis_enabled
                        and best_entropy > entropy_threshold
                        and state["total_loops_this_token"] == max_loops
                        and state["synthesis_events"] < max_synthesis_events
                    )

                    if should_synthesize:
                        state["synthesis_events"] += 1
                        print("    [Agentic ToT] Still unsatisfied! Initiating Test-Time Layer Synthesis...")
                        
                        # Check Cognitive Cache first. Cached entries are last-token deltas,
                        # so they can transfer across prompts with different sequence lengths.
                        cached_delta = (
                            _global_cache.retrieve(routed_h, verified_only=cache_verified_only)
                            if cache_enabled
                            else None
                        )
                        delta_to_apply = cached_delta
                        synthesis_successful = cached_delta is not None
                        synthesis_reason = "cache_hit" if cached_delta is not None else "optimizer"
                        loss_history = []
                        
                        if cached_delta is None:
                            with torch.enable_grad():
                                # Create a trainable parameter (the "synthesized layer" / dynamic vector)
                                v = torch.zeros_like(routed_h[:, -1:, :], dtype=torch.float32, requires_grad=True)

                                # Use Adam optimizer
                                opt = torch.optim.Adam([v], lr=0.2)
                                
                                belief_vec_t = belief_vec.to(routed_h.device).to(torch.float32) if belief_vec is not None else None
                                
                                # Test-Time Layer Synthesis (TTT) with Dynamic Compute
                                max_ttt_steps = 500
                                min_loss_threshold = -1.0 # Depends on Truth Vector magnitude, but low is good
                                
                                for step in range(max_ttt_steps):
                                    opt.zero_grad()
                                    
                                    h_syn = _add_last_token_delta(routed_h, v)
                                    
                                    # Forward through remaining layers to get final state
                                    h_curr = h_syn
                                    for j in range(l_idx + 1, len(M.model.model.layers)):
                                        layer_kwargs = dict(kwargs)
                                        layer_kwargs["use_cache"] = False
                                        for cache_key in ("past_key_value", "past_key_values"):
                                            if cache_key in layer_kwargs:
                                                del layer_kwargs[cache_key]
                                        
                                        # Fix CUDA Illegal Memory Access in SDPA:
                                        # If seq_len == 1 (generation phase), we removed past_key_value, 
                                        # so we only have 1 query and 1 key. But attention_mask is sized for the full past sequence.
                                        # This mismatch crashes SDPA. Removing attention_mask defaults to unmasked (correct for 1x1).
                                        if h_curr.shape[1] == 1:
                                            if "attention_mask" in layer_kwargs:
                                                del layer_kwargs["attention_mask"]
                                            if "position_ids" in layer_kwargs and layer_kwargs["position_ids"].shape[1] > 1:
                                                layer_kwargs["position_ids"] = layer_kwargs["position_ids"][:, -1:]
                                            
                                        layer_out = M.model.model.layers[j](h_curr, *args[1:], **layer_kwargs)
                                        h_curr = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                                    
                                    h_31_last = h_curr[:, -1:, :]
                                    
                                    # Logit Entropy
                                    h_norm_syn = M.model.model.norm(h_31_last.to(M.model.dtype))
                                    logits_syn = M.model.lm_head(h_norm_syn)
                                    entropy_syn = _entropy_from_logits(logits_syn).mean()
                                    
                                    # Truth Projection
                                    if belief_vec_t is not None:
                                        proj_truth = (h_31_last.squeeze(-2).float() * belief_vec_t).sum(dim=-1).mean()
                                    else:
                                        proj_truth = torch.tensor(0.0, device=h_curr.device)
                                    
                                    # Joint Loss: Minimize Entropy, Maximize Truth, Penalize Norm
                                    lambd = 1.0 # Reduced from 2.0
                                    norm_penalty = 0.5 * torch.norm(v)
                                    loss = entropy_syn - lambd * proj_truth + norm_penalty
                                    
                                    loss.backward()
                                    opt.step()
                                    
                                    current_loss = loss.item()
                                    loss_history.append(current_loss)
                                    
                                    if step < 3 or step % 10 == 0:
                                        print(f"      [Synthesis Step {step+1}] Loss: {current_loss:.2f} | Entropy: {entropy_syn.item():.2f}")
                                    
                                    # Convergence Check
                                    if current_loss < min_loss_threshold:
                                        print(f"      [Synthesis Step {step+1}] Successfully synthesized layer! Loss: {current_loss:.2f}")
                                        synthesis_successful = True
                                        synthesis_reason = "loss_threshold"
                                        break
                                    
                                    # Dynamic Compute Allocation: Check for plateau
                                    if len(loss_history) > 10:
                                        recent_losses = loss_history[-10:]
                                        loss_diff = recent_losses[0] - recent_losses[-1] # positive if decreasing
                                        
                                        if loss_diff < 0.05: # Loss has plateaued (d(loss)/dt == 0)
                                            print(f"      [Synthesis Step {step+1}] Loss plateaued at {current_loss:.2f}. Model is mathematically trapped.")
                                            
                                            # Inject Humility Vector
                                            print("    [Agentic ToT] ULTIMATE HUMILITY: Model cannot deduce answer. Asking user for help.")
                                            if state["humility_vec"] is not None:
                                                humility_t = state["humility_vec"].to(routed_h.device).to(torch.float32)
                                                # Massive injection to force humility
                                                v.data = humility_t.view(1, 1, -1) * 20.0
                                                synthesis_reason = "humility_plateau"
                                            else:
                                                # Fallback
                                                noise = torch.randn_like(v) * routed_h.float().std() * 5.0
                                                v.data = noise
                                                synthesis_reason = "noise_plateau"
                                                
                                            synthesis_successful = True
                                            break
                                if synthesis_successful:
                                    delta_to_apply = v.detach()
                                
                            # Clean up VRAM after gradient operations. Cache hits skip this
                            # so retrieval does not pay optimizer cleanup latency.
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()

                        if synthesis_successful and delta_to_apply is not None and cached_delta is None:
                            metadata = {
                                "reason": synthesis_reason,
                                "start_layer": state["start_layer"],
                                "end_layer": l_idx,
                                "steps": len(loss_history),
                            }
                            if synthesis_recorder is not None:
                                synthesis_recorder.append(
                                    {
                                        "trigger": routed_h.detach().cpu(),
                                        "delta": delta_to_apply.detach().cpu(),
                                        "metadata": dict(metadata),
                                    }
                                )
                            if cache_write_enabled:
                                _global_cache.store(routed_h, delta_to_apply, metadata=metadata)

                        # Apply the fully synthesized vector to the hidden state permanently
                        if delta_to_apply is not None:
                            routed_h = _add_last_token_delta(routed_h, delta_to_apply.detach())
                    
                    if isinstance(out, tuple):
                        out = (routed_h,) + tuple(out[1:])
                    else:
                        out = routed_h
                        
                    h = routed_h
                    
            state["prev_h"] = h.detach().clone()
            
            return out
            
        return hook

    for l in range(M.n_layers):
        handles.append(M.model.model.layers[l].register_forward_hook(make_hook(l), with_kwargs=True))
        
    return handles

@torch.no_grad()
def generate_agentic_text(
    M,
    vecs,
    belief_vec=None,
    humility_vec=None,
    instruction="",
    alpha=15.0,
    max_new_tokens=64,
    epsilon=0.05,
    entropy_threshold=2.0,
    max_loops=3,
    force_synthesis=False,
    cache_enabled=True,
    cache_write_enabled=False,
    cache_verified_only=True,
    synthesis_enabled=True,
    max_synthesis_events=1,
    synthesis_recorder=None,
    max_tool_calls=4,
    max_routing_events=4,
):
    inputs = _inputs(M, instruction)
    original_plen = inputs["input_ids"].shape[1]
    
    handles = get_agentic_handles(
        M,
        vecs,
        belief_vec=belief_vec,
        humility_vec=humility_vec,
        alpha=alpha,
        epsilon=epsilon,
        entropy_threshold=entropy_threshold,
        max_loops=max_loops,
        force_synthesis=force_synthesis,
        cache_enabled=cache_enabled,
        cache_write_enabled=cache_write_enabled,
        cache_verified_only=cache_verified_only,
        synthesis_enabled=synthesis_enabled,
        max_synthesis_events=max_synthesis_events,
        synthesis_recorder=synthesis_recorder,
        max_routing_events=max_routing_events,
    )
    
    try:
        from transformers import StoppingCriteriaList
        from invariants.tool_utils import ToolStoppingCriteria, intercept_tool_call, evaluate_python_expression
        
        eos_ids = [M.tok.eos_token_id]
        if 128009 not in eos_ids:
            eos_ids.append(128009)
            
        current_inputs = {k: v.clone() for k, v in inputs.items()}
        tokens_generated = 0
        tool_calls = 0
        full = current_inputs["input_ids"]
        
        while tokens_generated < max_new_tokens:
            criteria = StoppingCriteriaList([ToolStoppingCriteria(M.tok, start_length=current_inputs["input_ids"].shape[1])])
            out = M.model.generate(
                **current_inputs, 
                max_new_tokens=max_new_tokens - tokens_generated, 
                do_sample=False,
                use_cache=True, 
                pad_token_id=M.tok.eos_token_id,
                eos_token_id=eos_ids,
                stopping_criteria=criteria
            )
            
            plen = current_inputs["input_ids"].shape[1]
            new_tokens = out[0][plen:]
            tokens_generated += len(new_tokens)
            
            decoded = M.tok.decode(new_tokens, skip_special_tokens=True)
            expr = intercept_tool_call(decoded)
            
            if expr:
                tool_calls += 1
                if tool_calls > max_tool_calls:
                    full = out
                    break
                result = evaluate_python_expression(expr)
                result_str = f" = {result}\n"
                result_ids = M.tok.encode(result_str, add_special_tokens=False, return_tensors="pt").to(out.device)
                new_input_ids = torch.cat([out, result_ids], dim=1)
                new_attn_mask = torch.ones(new_input_ids.shape, dtype=torch.long, device=out.device)
                current_inputs = {"input_ids": new_input_ids, "attention_mask": new_attn_mask}
                full = new_input_ids
            else:
                full = out
                break
            
            # Note: We continue the loop and will run M.model.generate again with the new context!
            # The hooks in `handles` are still active and will intercept this new generation pass!
    finally:
        for h in handles:
            h.remove()
            
    return M.tok.decode(full[0, original_plen:], skip_special_tokens=True).strip()

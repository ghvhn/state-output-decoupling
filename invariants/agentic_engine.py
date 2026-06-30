import time
import torch
import torch.nn.functional as F
from pathlib import Path

from invariants.engine import _ended_with_eos, _generated_text_satisfies_stop, _inputs
from invariants.social_hunt import get_steer_vector
from invariants.cognitive_cache import CognitiveCache

class NeedsDisambiguationError(Exception):
    pass


class GenerationBudgetExceeded(Exception):
    pass


def _raise_if_generation_deadline_expired(config):
    deadline = getattr(config, "_generation_deadline", None)
    if deadline is not None and time.time() >= deadline:
        raise GenerationBudgetExceeded("generation_budget_exceeded")


# Global cache instance
_global_cache = CognitiveCache()

_vector_cache = {}
def _get_vector(name, device):
    if name not in _vector_cache:
        path = Path(__file__).parent / f"{name}.pt"
        if path.exists():
            _vector_cache[name] = torch.load(path, map_location=device)
        else:
            _vector_cache[name] = None
    vec = _vector_cache[name]
    if vec is None:
        return None
    if isinstance(vec, dict):
        return {k: v.to(device) for k, v in vec.items()}
    return vec.to(device)

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


def _layer_vector(name, layer_index, device):
    vec = _get_vector(name, device)
    if isinstance(vec, dict):
        vec = vec.get(layer_index)
    return None if vec is None else vec.to(device)


def _last_token_matrix(t):
    t = t.float()
    if t.ndim == 1:
        return t.view(1, -1)
    if t.ndim == 2:
        return t[-1:, :]
    return t.reshape(-1, t.shape[-2], t.shape[-1])[:, -1, :]


def _generation_time_pressure(config):
    deadline = getattr(config, "_generation_deadline", None)
    budget = getattr(config, "_generation_budget_sec", None)
    if deadline is None or budget is None or budget <= 0:
        return 0.0
    remaining = max(0.0, deadline - time.time())
    elapsed = max(0.0, budget - remaining)
    return max(0.0, min(1.0, elapsed / budget))


def _maybe_apply_time_gated_urgency(h_syn, layer_index, config, state):
    urgency_vec = _layer_vector("urgency_vector", layer_index, h_syn.device)
    if urgency_vec is None:
        return h_syn

    if getattr(config, "continuous_urgency_injection", False):
        coef = float(getattr(config, "urgency_max_coefficient", 0.8))
        norm = urgency_vec.float().norm().clamp_min(1e-6)
        state["last_time_awareness"] = {
            "mode": "continuous_urgency",
            "coefficient": coef,
        }
        return _add_last_token_delta(h_syn, coef * urgency_vec.float() / norm)

    if not getattr(config, "time_awareness_gated_urgency", True):
        return h_syn

    time_vec = _layer_vector("time_awareness_vector", layer_index, h_syn.device)
    if time_vec is None:
        return h_syn

    pressure = _generation_time_pressure(config)
    if pressure <= 0:
        return h_syn

    h_last = _last_token_matrix(h_syn)
    t_last = _last_token_matrix(time_vec)
    sim = F.cosine_similarity(h_last, t_last.expand_as(h_last), dim=-1).mean().item()
    threshold = float(getattr(config, "time_awareness_threshold", 0.45))
    state["last_time_awareness"] = {
        "mode": "time_awareness_gated",
        "similarity": sim,
        "threshold": threshold,
        "time_pressure": pressure,
        "applied": False,
    }
    if sim < threshold:
        return h_syn

    gate = max(0.0, min(1.0, (sim - threshold) / max(1e-6, 1.0 - threshold)))
    coef = float(getattr(config, "urgency_max_coefficient", 0.8)) * pressure * gate
    if coef <= 0:
        return h_syn

    norm = urgency_vec.float().norm().clamp_min(1e-6)
    state["last_time_awareness"].update(
        {
            "gate": gate,
            "coefficient": coef,
            "applied": True,
        }
    )
    return _add_last_token_delta(h_syn, coef * urgency_vec.float() / norm)



def get_agentic_handles(
    M,
    vecs=None,
    belief_vec=None,
    humility_vec=None,
    config=None,
    synthesis_recorder=None,
):
    """
    Parallel Latent Search (ToT)
    Dynamically clones the hidden state into 3 branches at the plateau.
    Injects 3 different optimizers. Evaluates entropy. Keeps the best.
    """
    from invariants.config import AgenticConfig, _global_registry
    if config is None:
        config = AgenticConfig()
    if vecs is None:
        vecs = _global_registry.get_vecs(M)
    if belief_vec is None:
        belief_vec = _global_registry.get_special_vec("belief_vector")
    if humility_vec is None:
        humility_vec = _global_registry.get_special_vec("humility_vector")
        
    alpha = config.alpha
    epsilon = config.epsilon
    entropy_threshold = config.entropy_threshold
    max_loops = config.max_loops
    force_synthesis = config.force_synthesis
    cache_enabled = config.cache_enabled
    cache_write_enabled = config.cache_write_enabled
    cache_verified_only = config.cache_verified_only
    ignore_oracle_cache = config.ignore_oracle_cache
    excluded_question_key = (
        config.benchmark_question_key
        if getattr(config, "exclude_same_question_cache", False)
        else None
    )
    excluded_oracle_question_key = (
        config.benchmark_question_key
        if config.exclude_same_question_oracle_cache
        else None
    )
    synthesis_enabled = config.synthesis_enabled
    max_synthesis_events = config.max_synthesis_events
    max_synthesis_steps = max(0, int(getattr(config, "max_synthesis_steps", 60)))
    use_tuned_lens = bool(getattr(config, "use_tuned_lens", False))
    if not hasattr(config, "_synthesis_events_used"):
        config._synthesis_events_used = 0
    max_routing_events = config.max_routing_events
    interactive_disambiguation = config.interactive_disambiguation
    
    if use_tuned_lens and not hasattr(M, "_tuned_lens"):
        from pathlib import Path
        out_dir = Path(__file__).parent / "out"
        configured_lens = getattr(config, "tuned_lens_path", None)
        lens_pt = Path(configured_lens) if configured_lens else out_dir / "lens_native.pt"
        if lens_pt.exists():
            print(f"    [Agentic Engine] Loading Tuned Lens from {lens_pt}...", flush=True)
            import torch.nn as nn
            d = M.d_model
            nL = M.n_layers
            tr = nn.ModuleList([nn.Linear(d, d, bias=True) for _ in range(nL - 1)]).to(M.device)
            state_dict = torch.load(lens_pt, map_location=M.device)
            tr.load_state_dict(state_dict["state"])
            M._tuned_lens = tr
        else:
            print("    [Agentic Engine] WARNING: lens_native.pt not found. Falling back to exact synthesis.", flush=True)
            M._tuned_lens = None

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
        "synthesis_recorder": synthesis_recorder,
    }
    
    def make_hook(l_idx):
        def hook(module, args, kwargs, out):
            _raise_if_generation_deadline_expired(config)
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
                    winner_name = state["branch_names"][best_idx]
                    
                    if "synthesis_recorder" in state and state["synthesis_recorder"] is not None:
                        state["synthesis_recorder"].append({
                            "type": "routing_trace",
                            "loop": state["total_loops_this_token"],
                            "entropies": {
                                state["branch_names"][0]: entropies[0].item(),
                                state["branch_names"][1]: entropies[1].item(),
                                state["branch_names"][2]: entropies[2].item(),
                            },
                            "winner": winner_name,
                            "best_entropy": best_entropy
                        })
                    
                    print(f"    [Agentic ToT] Token Loop {state['total_loops_this_token']} | "
                          f"Soc: {entropies[0]:.2f}, Cre: {entropies[1]:.2f}, Ana: {entropies[2]:.2f} "
                          f"-> WINNER: {winner_name} (Entropy: {best_entropy:.2f})")
                    
                    # Collapse back to batch=1 with the winning state
                    routed_h = h_parallel[best_idx].unsqueeze(0)
                    
                    # USER REQUEST: "If the model isn't satisfied, it shouldn't be confident in its answer"
                    # UPDATE: "but this learning has to be influenced by the model's internal skepticism."
                    # Dynamic Test-Time Layer Synthesis
                    should_synthesize = (
                        synthesis_enabled
                        and best_entropy > entropy_threshold
                        and state["synthesis_events"] < max_synthesis_events
                        and config._synthesis_events_used < max_synthesis_events
                        and max_synthesis_steps > 0
                    )

                    if should_synthesize:
                        state["synthesis_events"] += 1
                        config._synthesis_events_used += 1
                        print("    [Agentic ToT] Still unsatisfied! Initiating Test-Time Layer Synthesis...")
                        
                        # Check Cognitive Cache first. Cached entries are last-token deltas,
                        # so they can transfer across prompts with different sequence lengths.
                        cached_delta = (
                            _global_cache.retrieve(
                                routed_h,
                                verified_only=cache_verified_only,
                                ignore_oracle_cache=ignore_oracle_cache,
                                excluded_question_key=excluded_question_key,
                                excluded_oracle_question_key=excluded_oracle_question_key,
                            )
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
                                max_ttt_steps = max_synthesis_steps
                                min_loss_threshold = -1.0 # Depends on Truth Vector magnitude, but low is good
                                
                                for step in range(max_ttt_steps):
                                    _raise_if_generation_deadline_expired(config)
                                    opt.zero_grad()
                                    
                                    h_syn = _add_last_token_delta(routed_h, v)
                                    
                                    h_syn = _maybe_apply_time_gated_urgency(h_syn, l_idx, config, state)
                                    
                                    # Use Tuned Lens if available to bypass the rest of the stack
                                    if (
                                        use_tuned_lens
                                        and hasattr(M, "_tuned_lens")
                                        and M._tuned_lens is not None
                                        and l_idx < len(M._tuned_lens)
                                    ):
                                        h_curr = h_syn + M._tuned_lens[l_idx](h_syn.float()).to(h_syn.dtype)
                                    else:
                                        # Fallback to the slow, exact computation
                                        h_curr = h_syn
                                        for j in range(l_idx + 1, len(M.model.model.layers)):
                                            layer_kwargs = dict(kwargs)
                                            layer_kwargs["use_cache"] = False
                                            for cache_key in ("past_key_value", "past_key_values"):
                                                if cache_key in layer_kwargs:
                                                    del layer_kwargs[cache_key]
                                            
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
                                            
                                            try:
                                                h_target = routed_h[:, -1:, :].float().squeeze(0)
                                                
                                                phenomenality = {}
                                                
                                                v_target = v.detach().float().squeeze(0)
                                                
                                                def vector_similarity(vector_name: str):
                                                    vec = _get_vector(vector_name, v_target.device)
                                                    if isinstance(vec, dict):
                                                        vec = vec.get(l_idx)
                                                    if vec is None:
                                                        return None
                                                    return F.cosine_similarity(
                                                        v_target,
                                                        vec.float().squeeze(0),
                                                        dim=-1,
                                                    ).item()

                                                monitored_vectors = {
                                                    "ambiguity": ("ambiguity_vector", True, "Ambiguity detected."),
                                                    "repetition": ("repetition_vector", True, "Repetition detected."),
                                                    "disagreement": ("disagreement_vector", True, "Disagreement detected."),
                                                    "time_awareness": ("time_awareness_vector", False, None),
                                                    "validated_flow": ("validated_flow_vector", False, None),
                                                    "needless_interrupt": ("needless_interrupt_vector", False, None),
                                                    "narrowing_in": ("narrowing_in_vector", False, None),
                                                    "self_referential_momentum": ("self_referential_momentum_vector", False, None),
                                                    "warranted_confidence_legacy": ("warranted_confidence_vector", False, None),
                                                    "unwarranted_confidence_legacy": ("unwarranted_confidence_vector", False, None),
                                                }
                                                for metric, (vector_name, can_interrupt, message) in monitored_vectors.items():
                                                    sim = vector_similarity(vector_name)
                                                    if sim is None:
                                                        continue
                                                    phenomenality[metric] = sim
                                                    if can_interrupt and abs(sim) > 0.1:
                                                        raise NeedsDisambiguationError(message)
                                                
                                                state["last_phenomenality"] = phenomenality
                                                print(
                                                    "      [Phenomenality Log] "
                                                    f"Ambiguity: {phenomenality.get('ambiguity', 0):.2f} | "
                                                    f"Repetition: {phenomenality.get('repetition', 0):.2f} | "
                                                    f"Disagreement: {phenomenality.get('disagreement', 0):.2f} | "
                                                    f"Flow: {phenomenality.get('validated_flow', 0):.2f} | "
                                                    f"Interrupt: {phenomenality.get('needless_interrupt', 0):.2f}"
                                                )
                                            except NeedsDisambiguationError:
                                                raise
                                            except Exception as e:
                                                pass

                                            print("    [Agentic ToT] Model is mathematically trapped. Injecting organic self-correction vector!")
                                            try:
                                                if not hasattr(config, "organic_correction_vector"):
                                                    config.organic_correction_vector = torch.load("invariants/organic_correction_vector.pt", map_location=routed_h.device)
                                                
                                                # The extracted organic vector is the mean shift of successful corrections
                                                organic_delta = config.organic_correction_vector.unsqueeze(0).unsqueeze(0).to(routed_h.dtype)
                                                
                                                # Scale factor to amplify the correction (empirically tuned, 1.5x)
                                                delta_to_apply = organic_delta * 1.5
                                                
                                                synthesis_successful = True
                                                break
                                            except Exception as e:
                                                print(f"    [Agentic ToT] Failed to inject organic vector: {e}")
                                                synthesis_successful = False
                                                break
                                if synthesis_successful and delta_to_apply is None:
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
                                "expert": state["branch_names"][best_idx],
                                "phenomenality": state.get("last_phenomenality", {}),
                                "time_awareness": state.get("last_time_awareness", {}),
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
        
    return handles, state

@torch.no_grad()
def generate_agentic_text(
    M,
    *positional,
    instruction="",
    vecs=None,
    belief_vec=None,
    humility_vec=None,
    config=None,
    max_new_tokens=None,
    synthesis_recorder=None,
    chatty_log=False,
    max_tool_calls=None,
    stop_after_final_answer=False,
    stop_after_verifier_answer=False,
    max_time=None,
    confidence_stabilization=False,
    **legacy_overrides,
):
    from invariants.config import AgenticConfig
    if config is None:
        config = AgenticConfig()

    for arg in positional:
        if isinstance(arg, dict):
            if vecs is not None:
                raise TypeError("generate_agentic_text received vecs more than once.")
            vecs = arg
        elif instruction:
            raise TypeError("generate_agentic_text received instruction more than once.")
        else:
            instruction = arg

    legacy_aliases = {
        "allow_synthesis": "synthesis_enabled",
    }
    config_fields = set(getattr(config, "__dataclass_fields__", {}).keys())
    for key, value in list(legacy_overrides.items()):
        target = legacy_aliases.get(key, key)
        if target in config_fields:
            setattr(config, target, value)
            legacy_overrides.pop(key)
    if legacy_overrides:
        unknown = ", ".join(sorted(legacy_overrides))
        raise TypeError(f"Unknown generate_agentic_text options: {unknown}")
    if chatty_log:
        config.chatty_log = True

    inputs = _inputs(M, instruction)
    original_plen = inputs["input_ids"].shape[1]
    
    # We still allow max_new_tokens override here because it's per-generation
    if max_new_tokens is None:
        max_new_tokens = config.max_new_tokens
    if max_tool_calls is None:
        max_tool_calls = getattr(config, "max_tool_calls", 8)

    previous_deadline = getattr(config, "_generation_deadline", None)
    previous_budget = getattr(config, "_generation_budget_sec", None)
    if max_time is not None and max_time > 0:
        config._generation_deadline = time.time() + float(max_time)
        config._generation_budget_sec = float(max_time)
    else:
        config._generation_deadline = None
        config._generation_budget_sec = None
    
    handles, state = get_agentic_handles(
        M,
        vecs=vecs,
        belief_vec=belief_vec,
        humility_vec=humility_vec,
        config=config,
        synthesis_recorder=synthesis_recorder,
    )
    state["confidence_stabilization"] = confidence_stabilization
    
    try:
        from transformers import StoppingCriteriaList, LogitsProcessorList, LogitsProcessor
        from invariants.tool_utils import (
            FinalAnswerStoppingCriteria,
            TimeStoppingCriteria,
            ToolStoppingCriteria,
            VerifierStoppingCriteria,
            iter_tool_calls,
            evaluate_python_expression,
        )
        
        class ContextTracker(LogitsProcessor):
            def __call__(self, input_ids, scores):
                state["current_input_ids"] = input_ids
                return scores
                
        state["tokenizer"] = M.tok
        state["current_input_ids"] = inputs["input_ids"]
        
        eos_ids = [M.tok.eos_token_id]
        if 128009 not in eos_ids:
            eos_ids.append(128009)
            
        current_inputs = {k: v.clone() for k, v in inputs.items()}
        tokens_generated = 0
        tool_calls = 0
        processed_tool_ends: set[int] = set()
        full = current_inputs["input_ids"]
        generation_started = time.time()
        
        while tokens_generated < max_new_tokens:
            remaining_time = None
            if max_time is not None and max_time > 0:
                remaining_time = max_time - (time.time() - generation_started)
                if remaining_time <= 0:
                    break
            chunk_tokens = min(max_new_tokens - tokens_generated, 24)
            start_length = current_inputs["input_ids"].shape[1]
            stopping_criteria = [ToolStoppingCriteria(M.tok, start_length=start_length)]
            if stop_after_final_answer:
                stopping_criteria.append(FinalAnswerStoppingCriteria(M.tok, start_length=start_length))
            if stop_after_verifier_answer:
                stopping_criteria.append(VerifierStoppingCriteria(M.tok, start_length=start_length))
            if remaining_time is not None:
                stopping_criteria.append(TimeStoppingCriteria(time.time() + remaining_time))
            criteria = StoppingCriteriaList(stopping_criteria)
            processors = LogitsProcessorList([ContextTracker()])
            
            try:
                generate_kwargs = {
                    **current_inputs,
                    "max_new_tokens": chunk_tokens,
                    "do_sample": False,
                    "use_cache": True,
                    "pad_token_id": M.tok.eos_token_id,
                    "eos_token_id": eos_ids,
                    "stopping_criteria": criteria,
                    "logits_processor": processors,
                }
                if remaining_time is not None:
                    generate_kwargs["max_time"] = remaining_time
                out = M.model.generate(**generate_kwargs)
            except GenerationBudgetExceeded:
                print("    [Agentic ToT] Generation budget reached; returning partial output.", flush=True)
                full = current_inputs["input_ids"]
                break
            except NeedsDisambiguationError as e:
                err_msg = str(e)
                print(f"\n    [Interlocutor] Cognitive Probe Matched: {err_msg} Formulating resolving question...")
                
                # Disable hooks for the sub-generation
                for h in handles:
                    h.remove()
                    
                q_prompt = (
                    f"You are a mathematical reasoning engine. You are stuck on the following problem due to {err_msg.lower()}\n"
                    f"{instruction}\n\n"
                    "Ask the user a single, highly specific clarifying question to resolve it."
                )
                q_inputs = _inputs(M, q_prompt)
                q_out = M.model.generate(**q_inputs, max_new_tokens=500, do_sample=True, temperature=0.7, pad_token_id=M.tok.eos_token_id)
                question = M.tok.decode(q_out[0][q_inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                
                print(f"\n[Model Question] {question}")

                event = {"reason": err_msg, "question": question}
                if hasattr(config, "clarifying_questions"):
                    config.clarifying_questions.append(event)

                if config.defer_disambiguation:
                    print("[Deferred Clarification] Recording question and returning control to the benchmark.")
                    raise NeedsDisambiguationError(question)

                if config.interactive_disambiguation:
                    from invariants.tool_utils import popup_massive_question
                    popup_massive_question(question)
                    user_answer = input("[Your Answer] (Press Enter for 'Enough information is present in the question'): ").strip()
                    if not user_answer:
                        user_answer = config.clarification_fallback or "Enough information is present in the question."
                    clarification_label = "Human Clarification"
                else:
                    user_answer = (
                        config.clarification_fallback
                        or "Resolve the uncertainty internally and continue without external information."
                    )
                    clarification_label = "Internal Disambiguation Policy"
                    print(f"[Internal Disambiguation Policy]: {user_answer}")
                
                clarification = f"\n\n[{clarification_label}]: {user_answer}\n"
                clarification_ids = M.tok.encode(clarification, add_special_tokens=False, return_tensors="pt").to(current_inputs["input_ids"].device)
                
                new_input_ids = torch.cat([current_inputs["input_ids"], clarification_ids], dim=1)
                new_attn_mask = torch.ones(new_input_ids.shape, dtype=torch.long, device=new_input_ids.device)
                current_inputs = {"input_ids": new_input_ids, "attention_mask": new_attn_mask}
                
                # Disable cache writes to avoid corrupting vectors with human text
                config.cache_write_enabled = False
                
                # Re-attach hooks
                handles, state = get_agentic_handles(
                    M, vecs=vecs, belief_vec=belief_vec, humility_vec=humility_vec,
                    config=config, synthesis_recorder=synthesis_recorder
                )
                
                state["tokenizer"] = M.tok
                state["current_input_ids"] = current_inputs["input_ids"]
                continue
                
            plen = current_inputs["input_ids"].shape[1]
            new_tokens = out[0][plen:]
            if new_tokens.numel() == 0:
                full = out
                break
            tokens_generated += len(new_tokens)
            
            if config.chatty_log:
                chunk = M.tok.decode(new_tokens, skip_special_tokens=True)
                if chunk.strip():
                    print(f"    [Agentic ToT] Token Chunk generated: {repr(chunk)}")
            
            current_inputs = {"input_ids": out, "attention_mask": torch.ones(out.shape, dtype=torch.long, device=out.device)}
            generated_text = M.tok.decode(out[0, original_plen:], skip_special_tokens=True)
            tool_call = next(
                (
                    (end, expr)
                    for _, end, expr in iter_tool_calls(generated_text)
                    if end not in processed_tool_ends
                ),
                None,
            )
            expr = None if tool_call is None else tool_call[1]
            
            if expr:
                processed_tool_ends.add(tool_call[0])
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
                if (
                    _ended_with_eos(new_tokens, eos_ids)
                    or _generated_text_satisfies_stop(
                        generated_text,
                        stop_after_final_answer=stop_after_final_answer,
                        stop_after_verifier_answer=stop_after_verifier_answer,
                    )
                    or len(new_tokens) < chunk_tokens
                ):
                    break
            
            # Note: We continue the loop and will run M.model.generate again with the new context!
            # The hooks in `handles` are still active and will intercept this new generation pass!
    finally:
        for h in handles:
            h.remove()
        config._generation_deadline = previous_deadline
        config._generation_budget_sec = previous_budget
             
    return M.tok.decode(full[0, original_plen:], skip_special_tokens=True).strip()

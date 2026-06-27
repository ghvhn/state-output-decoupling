import torch
import torch.nn.functional as F
import time
from invariants.engine import _inputs, _generate_ids

def get_recurrent_handles(M, epsilon=0.05, entropy_threshold=2.0, max_loops=3):
    """
    Hooks layers to dynamically detect the logic plateau and loop recurrently if
    uncertainty (entropy) is high when exiting the plateau.
    """
    handles = []
    
    # We maintain a shared state across all layer hooks
    state = {
        "prev_h": None,
        "start_layer": -1,
        "end_layer": -1,
        "loop_count": 0,
        "total_loops_this_token": 0,
    }
    
    def make_hook(l_idx):
        def hook(module, args, kwargs, out):
            # Extract hidden states
            if isinstance(out, tuple):
                h = out[0]
            else:
                h = out
                
            # Reset state on layer 0 (start of a new forward pass for a new token)
            if l_idx == 0:
                state["prev_h"] = h.detach().clone()
                state["start_layer"] = -1
                state["end_layer"] = -1
                state["total_loops_this_token"] = 0
                return out
                
            prev_h = state["prev_h"]
            
            # Compute velocity per-token
            curr_token_h = h.float()
            prev_token_h = prev_h.float()
            
            cos_sim = F.cosine_similarity(curr_token_h, prev_token_h, dim=-1)
            velocity = 1.0 - cos_sim  # [batch, seq]
            
            # We look at the velocity of the LAST token for dynamic routing decisions
            last_token_velocity = velocity[:, -1].mean().item()
            
            if last_token_velocity < epsilon and state["start_layer"] == -1:
                state["start_layer"] = l_idx
                
            # If we exit the plateau
            if state["start_layer"] != -1 and last_token_velocity > epsilon and state["end_layer"] == -1:
                state["end_layer"] = l_idx
                
                # We have reached the post-hoc boundary! 
                # Let's project the current state through the unembedding head
                # We only want to look at the last token
                h_last = h[:, -1:, :]
                h_norm = M.model.model.norm(h_last.to(M.model.dtype))
                logits = M.model.lm_head(h_norm)
                
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()
                
                # Check if we should loop
                if entropy > entropy_threshold and state["total_loops_this_token"] < max_loops:
                    state["total_loops_this_token"] += 1
                    
                    # Log the loop! (Optional, but good for visibility)
                    # print(f"    [Recurrent] High Entropy ({entropy:.2f}) at L{l_idx}. Looping to L{state['start_layer']}! (Loop {state['total_loops_this_token']})")
                    
                    # Extract kwargs needed to run the intermediate layers again manually
                    attention_mask = kwargs.get("attention_mask")
                    position_ids = kwargs.get("position_ids")
                    
                    # Manually pass the hidden state back through the plateau layers
                    routed_h = h
                    for i in range(state["start_layer"], l_idx + 1):
                        # We pass use_cache=False to prevent corrupting the global KV cache during the latent loop
                        layer_kwargs = dict(kwargs)
                        layer_kwargs["use_cache"] = False
                        if "past_key_value" in layer_kwargs:
                            del layer_kwargs["past_key_value"]
                        
                        layer_out = M.model.model.layers[i](routed_h, *args[1:], **layer_kwargs)
                        routed_h = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                    
                    # We have completed the loop. We will replace the original layer output with our routed_h
                    if isinstance(out, tuple):
                        out = (routed_h,) + tuple(out[1:])
                    else:
                        out = routed_h
                        
                    # Update h so prev_h is correct for the next layer in the global sequence
                    h = routed_h
                    
            state["prev_h"] = h.detach().clone()
            
            return out
            
        return hook

    for l in range(M.n_layers):
        handles.append(M.model.model.layers[l].register_forward_hook(make_hook(l), with_kwargs=True))
        
    return handles

@torch.no_grad()
def generate_recurrent_text(M, instruction, max_new_tokens=32, epsilon=0.05, entropy_threshold=2.0, max_loops=3):
    inputs = _inputs(M, instruction)
    plen = inputs["input_ids"].shape[1]
    
    handles = get_recurrent_handles(M, epsilon, entropy_threshold, max_loops)
    
    try:
        full = M.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            use_cache=True, pad_token_id=M.tok.eos_token_id,
        )
    finally:
        for h in handles:
            h.remove()
            
    return M.tok.decode(full[0, plen:], skip_special_tokens=True).strip()

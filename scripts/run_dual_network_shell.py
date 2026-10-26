"""Interactive Dual-Network Shell (CEN + DMN + Shared Substrate).

Provides an interactive console environment coupling the Central Executive Network
(the waking talker/doer) and the Default Mode Network (the reflective consolidator)
over Shared Substrate Memory. Supports live model generation or synthetic probe simulation.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import colorama
from colorama import Fore, Style
import torch

from invariants.central_executive import (
    CentralExecutiveNetwork,
    DefaultModeNetwork,
    SharedSubstrateMemory,
    ProbeExecutiveAdapter,
)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

colorama.init(autoreset=True)


def print_banner():
    print(Fore.CYAN + Style.BRIGHT + "=" * 70)
    print(Fore.CYAN + Style.BRIGHT + "   DUAL-NETWORK INTERACTIVE SHELL (CEN + DMN + SUBSTRATE)   ")
    print(Fore.CYAN + Style.BRIGHT + "=" * 70)
    print(Fore.WHITE + "  * Central Executive Network (CEN): Tracks activation derivatives & friction")
    print(Fore.WHITE + "  * Default Mode Network (DMN): Background consolidation & coherence")
    print(Fore.WHITE + "  * Type any prompt to generate a turn and observe cognitive friction")
    print(Fore.WHITE + "  * Commands: :status, :episodes, :gamma <val>, :probes, exit")
    print(Fore.CYAN + "=" * 70 + "\n")


def load_probes() -> dict:
    """Load default repository probes."""
    probe_files = {
        "ambiguity": ROOT / "invariants" / "ambiguity_vector.pt",
        "disagreement": ROOT / "invariants" / "disagreement_vector.pt",
        "validated_flow": ROOT / "invariants" / "validated_flow_vector.pt",
        "warranted_confidence": ROOT / "invariants" / "warranted_confidence_vector.pt",
        "urgency": ROOT / "invariants" / "urgency_vector.pt",
    }
    loaded = {}
    for name, path in probe_files.items():
        if path.exists():
            try:
                loaded[name] = torch.load(path, map_location="cpu")
            except Exception as e:
                print(Fore.YELLOW + f"Warning: Could not load {name}: {e}")
    return loaded


def run_interactive_dual_network():
    print_banner()

    # 1. Initialize Substrate & Networks
    substrate = SharedSubstrateMemory(buffer_size=500)
    cen = CentralExecutiveNetwork(substrate=substrate, gamma=0.08)
    dmn = DefaultModeNetwork(
        substrate=substrate,
        consolidation_window=4,
        friction_threshold=0.035,
        poll_timeout=0.2,
    )

    # 2. Load Probes and bind Adapter
    probes = load_probes()
    layers = [12, 14, 16, 18, 20]
    adapter = ProbeExecutiveAdapter(cen=cen, probes=probes, layers=layers)
    print(Fore.GREEN + f"[Init] Substrate online. Bound to Layers {layers} probes: {adapter.probe_names}")

    # 3. Start DMN background thread
    dmn.start()
    print(Fore.GREEN + "[Init] Default Mode Network (DMN) thread active in background.\n")

    # Optional model loading
    use_model = False
    model = None
    if "--model" in sys.argv or "-m" in sys.argv:
        try:
            from invariants.engine import load_model
            model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            for i, arg in enumerate(sys.argv):
                if arg in ("--model", "-m") and i + 1 < len(sys.argv):
                    model_name = sys.argv[i + 1]
            print(Fore.YELLOW + f"[System] Loading model {model_name}...")
            model = load_model(model_name)
            use_model = True
            print(Fore.GREEN + "[System] Model loaded successfully.")
        except Exception as e:
            print(Fore.RED + f"[System] Model load failed: {e}. Falling back to simulation mode.")

    try:
        while True:
            try:
                user_input = input(Fore.YELLOW + "DualNetwork > " + Style.RESET_ALL).strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting Dual-Network Shell...")
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                print("Closing session.")
                break

            if user_input.startswith(":"):
                parts = user_input.split()
                cmd = parts[0].lower()

                if cmd in (":status", ":cen", ":telemetry"):
                    tel = substrate.get_telemetry()
                    print(Fore.CYAN + "\n[Telemetry Dashboard]")
                    print(Fore.CYAN + f"  Snapshots Emitted: {tel['total_emitted']} | Consumed: {tel['total_consumed']}")
                    print(Fore.CYAN + f"  Buffer Depth:      {tel['buffer_depth']} | Dropped: {tel['dropped_snapshots']}")
                    print(Fore.CYAN + f"  Recent Friction:   {tel['average_recent_friction']:.6f} (gamma={cen.gamma})")
                    print(Fore.CYAN + f"  Consolidated EPs:  {tel['consolidated_episodes_count']}\n")
                    continue

                elif cmd in (":episodes", ":dmn", ":repo"):
                    print(Fore.CYAN + f"\n[DMN Consolidated Repository] ({len(substrate.consolidated_repository)} episodes):")
                    if not substrate.consolidated_repository:
                        print(Fore.CYAN + "  (No episodes consolidated yet. Generate more turns!)")
                    for ep in substrate.consolidated_repository[-8:]:
                        zone_color = Fore.RED if ep["state_zone"] == "high_friction_boundary" else Fore.GREEN
                        print(
                            Fore.CYAN + f"  {ep['episode_id']} | Steps {ep['start_step']:>2}..{ep['end_step']:>2} | "
                            + zone_color + f"Zone: {ep['state_zone']:<22}" + Fore.CYAN
                            + f" | Friction: {ep['mean_friction']:.5f} | Coherence: {ep['trajectory_coherence']:+.3f}"
                        )
                    print()
                    continue

                elif cmd == ":gamma":
                    if len(parts) > 1:
                        try:
                            cen.gamma = float(parts[1])
                            print(Fore.GREEN + f"[CEN] Gamma friction coefficient set to {cen.gamma}\n")
                        except ValueError:
                            print(Fore.RED + "Invalid gamma float value.\n")
                    else:
                        print(Fore.CYAN + f"[CEN] Current gamma: {cen.gamma}\n")
                    continue

                elif cmd == ":probes":
                    print(Fore.CYAN + f"\n[Active Concept Probes] Layer {layer}:")
                    for p in adapter.probe_names:
                        print(Fore.CYAN + f"  - {p}")
                    print()
                    continue

                elif cmd in (":remember", ":rem"):
                    rem_arg = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
                    if not rem_arg or rem_arg.lower() in ("thoughts", "turn", "last"):
                        while not substrate.activation_buffer.empty():
                            try:
                                s = substrate.activation_buffer.get_nowait()
                                dmn.window_buffer.append(s)
                            except Exception:
                                break
                        if dmn.window_buffer:
                            ep = dmn.consolidate_window(list(dmn.window_buffer), inflection_trigger=True)
                            dmn.window_buffer.clear()
                            print(Fore.GREEN + f"\n[Remember] Consolidated thoughts via CEN into {ep['episode_id']}:")
                            print(Fore.CYAN + f"  Friction: {ep['mean_friction']:.5f} | Coherence: {ep['trajectory_coherence']:+.3f} | Zone: {ep['state_zone']}\n")
                        elif substrate.consolidated_repository:
                            ep = substrate.consolidated_repository[-1]
                            print(Fore.GREEN + f"\n[Remember] Latest consolidated thought episode {ep['episode_id']}:")
                            print(Fore.CYAN + f"  Friction: {ep['mean_friction']:.5f} | Coherence: {ep['trajectory_coherence']:+.3f} | Zone: {ep['state_zone']}\n")
                        else:
                            print(Fore.YELLOW + "\n[Remember] No thought snapshots in buffer yet. Generate a turn first.\n")
                    else:
                        if use_model and model is not None:
                            from invariants.engine import _inputs, _hidden_states
                            inp = _inputs(model, rem_arg)
                            hs = _hidden_states(model, inp["input_ids"], inp.get("attention_mask"))
                            h_state = hs
                        else:
                            base_t = torch.randn(4096) + 1.5 * adapter.probe_vectors.get("validated_flow", torch.randn(4096))
                            h_state = {l: base_t + 0.1 * (l - adapter.layer) * torch.randn(4096) for l in adapter.layers}
                        snap = adapter.step_with_activation(
                            h_state,
                            token_string=rem_arg[:25],
                            extra_metadata={"source": "remember_command", "text": rem_arg[:100]},
                        )
                        print(Fore.GREEN + f"\n[Remember] Recorded to CEN trajectory buffer (Step {snap.step:02d}):")
                        print(Fore.CYAN + f"  \"{rem_arg}\" (friction={snap.friction_gradient:.6f})\n")
                    continue

                elif cmd in (":help", ":h"):
                    print(Fore.CYAN + "\nCommands:")
                    print("  :remember [thoughts|<text>] - Consolidate thoughts or store a fact via CEN")
                    print("  :status      - Show substrate buffer & friction telemetry")
                    print("  :episodes    - Show DMN consolidated episodic memory")
                    print("  :gamma <num> - Set friction sensitivity coefficient")
                    print("  :probes      - List active concept probes")
                    print("  exit / quit  - Exit shell\n")
                    continue

                else:
                    print(Fore.YELLOW + f"Unknown command: {cmd}. Type :help for commands.\n")
                    continue

            # Process User Thought / Turn
            print(Fore.WHITE + f"\n[CEN Thinking...] Processing input: \"{user_input}\"")

            if use_model and model is not None:
                # Real forward pass & generation with model
                from invariants.engine import _inputs, _hidden_states, generate_text
                try:
                    response = generate_text(model, user_input, max_new_tokens=64)
                except Exception as _gen_err:
                    response = f"(Generation error: {_gen_err})"

                print(Fore.GREEN + Style.BRIGHT + "\nMe: " + Style.RESET_ALL + response)

                inp = _inputs(model, response if response else user_input)
                h_all = _hidden_states(model, inp["input_ids"], inp.get("attention_mask"))
                snapshot = adapter.step_with_activation(
                    h_all,
                    token_string=(response if response else user_input)[:25],
                    extra_metadata={"user_prompt": user_input, "response": response},
                )
            else:
                # Simulated realistic semantic shift based on prompt sentiment/keywords across layers
                h_base = torch.randn(4096)
                lower_input = user_input.lower()

                # Synthesize concept alignment
                if any(w in lower_input for w in ("wait", "maybe", "confused", "uncertain", "?", "or")):
                    target_concept = "ambiguity"
                    response = f"I am sensing ambiguity in your thought ('{user_input}') and reflecting on potential clarifications."
                elif any(w in lower_input for w in ("no", "wrong", "contradict", "disagree", "error")):
                    target_concept = "disagreement"
                    response = f"I observe a cognitive divergence or disagreement around: '{user_input}'."
                elif any(w in lower_input for w in ("urgent", "hurry", "quick", "deadline", "fast")):
                    target_concept = "urgency"
                    response = f"Heightened urgency detected. Processing swiftly: '{user_input}'."
                else:
                    target_concept = "validated_flow"
                    response = f"I follow that line of inquiry clearly and maintain validated flow."

                # Construct layer-resolved simulated activations with depth gradient
                h_turn: Dict[int, torch.Tensor] = {}
                for l in adapter.layers:
                    depth_scale = 1.0 + 0.15 * (l - adapter.layer)
                    p_vec = adapter.layer_probe_vectors.get(target_concept, {}).get(l)
                    if p_vec is None:
                        p_vec = adapter.probe_vectors.get(target_concept, torch.randn(4096))
                    h_turn[l] = h_base + (2.0 * depth_scale) * p_vec

                print(Fore.GREEN + Style.BRIGHT + "\nMe: " + Style.RESET_ALL + response)

                snapshot = adapter.step_with_activation(
                    h_turn,
                    token_string=user_input[:25],
                    extra_metadata={"user_prompt": user_input, "response": response},
                )

            scores = snapshot.metadata.get("probe_scores", {})
            depth_grads = snapshot.metadata.get("depth_gradients", {})
            f_color = Fore.RED if snapshot.friction_gradient >= 0.035 else Fore.GREEN
            print(
                Fore.WHITE + f"  [Step {snapshot.step:02d}] "
                + f_color + f"Dynamic Friction: {snapshot.friction_gradient:.6f} (|d|={snapshot.derivative_magnitude:.4f}) "
                + Fore.WHITE + f"| J-Space: {[round(x, 3) for x in snapshot.j_space_variance]}"
            )
            grad_str = ""
            if depth_grads:
                active_grads = [f"d{k}/dl: {v:+.3f}" for k, v in depth_grads.items() if abs(v) > 0.001][:2]
                if active_grads:
                    grad_str = " | Depth: " + ", ".join(active_grads)
            print(
                Fore.CYAN + "  [Probes] "
                + " | ".join(f"{k}: {v:+.2f}" for k, v in list(scores.items())[:4])
                + grad_str
            )

            # Let DMN digest
            time.sleep(0.05)
            if substrate.consolidated_repository:
                latest_ep = substrate.consolidated_repository[-1]
                if latest_ep.get("end_step") == snapshot.step:
                    zone_c = Fore.RED if latest_ep["state_zone"] == "high_friction_boundary" else Fore.GREEN
                    print(
                        zone_c + f"  [DMN Event] Consolidated {latest_ep['episode_id']} ({latest_ep['state_zone']}) "
                        + f"- Trajectory Coherence: {latest_ep['trajectory_coherence']:+.3f}"
                    )
            print()

    finally:
        dmn.stop()
        if dmn.is_alive():
            dmn.join(timeout=1.0)
        print("Dual-network runtime stopped.")


if __name__ == "__main__":
    run_interactive_dual_network()

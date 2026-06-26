"""
Phase 1 — Signature Discovery Pipeline

Orchestrates the full run:
  1. Load model and conversations
  2. Replay conversations, capture residual stream on assistant turns
  3. Build point clouds per layer band
  4. Run TDA, compute fingerprints
  5. Run discovery or confirmation mode
  6. Save signature library

All data is namespaced by model name in both HDF5 and signatures.json,
so a single store directory can hold results from multiple models without collision.

All intermediate results are stored at each tier so downstream analysis
can be rerun without re-extracting activations from the model.
"""

import yaml
import json
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import extraction.model as M
from extraction.replay import replay, live as live_replay
import compute
from data.conversations import load as load_conversations, all_domains
from tda.cloud import build as build_cloud
from tda.compress import reduce as compress
from tda.homology import run as run_homology, persistent_features
from tda.fingerprint import from_diagrams, average
from tda.latent_variables import analyze as latent_variable_analysis
from tda.patterns import analyze_all_dimensions, cross_dimension_analysis
from domains.discovery import cluster, domain_signatures
from domains.confirmation import confirm
from store import activations as act_store
from store import signatures as sig_store


def load_config(path: str = "config/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _process_band(conv_id, band_name, combined, pca_dims, maxdim, metric, persistence_threshold):
    """
    Compute-only TDA worker — no I/O. Returns everything needed for the main
    process to write results. Keeping I/O out of workers lets us batch all
    36 jobs (12 convs × 3 bands) into one pool and use all 28 cores at once
    instead of 3 at a time.
    """
    from tda.cloud import build as build_cloud
    from tda.compress import reduce as compress
    from tda.homology import run as run_homology, persistent_features
    from tda.fingerprint import from_diagrams

    cloud = build_cloud(combined)
    compressed, _ = compress(cloud, pca_dims)
    diagrams = run_homology(compressed, maxdim=maxdim, metric=metric)
    diagrams = persistent_features(diagrams, threshold=persistence_threshold)
    fingerprint = from_diagrams(diagrams)
    return conv_id, band_name, diagrams, fingerprint


def _run_patterns(conv_id, band_name, combined, top_k=20, score_threshold=0.6):
    """
    Pattern analysis worker — runs in the same pool as TDA so CPU stays
    saturated while GPU is generating the next conversation.
    combined: [n_layers, n_tokens, d_model] float16
    Returns flagged dimensions and cross-dimension coupling.
    """
    from tda.patterns import analyze_all_dimensions, cross_dimension_analysis

    activation_matrix = combined.mean(axis=0).astype(np.float32)  # [n_tokens, d_model]
    flagged = analyze_all_dimensions(activation_matrix, top_k=top_k, score_threshold=score_threshold)
    flagged_dims = [r["dimension"] for r in flagged]
    coupling = cross_dimension_analysis(activation_matrix, flagged_dims) if len(flagged_dims) > 1 else []
    return conv_id, band_name, flagged, coupling


def run(config_path: str = "config/default.yaml", conversations_path: str = None):
    cfg = load_config(config_path)
    model_name = cfg["model"]["name"]

    compute.report()
    compute.set_pysr_parallelism()
    print(f"\nLoading model: {model_name}")
    model = M.load(model_name, cfg["model"]["device"])

    bands = {k: v for k, v in cfg["extraction"]["layer_bands"].items()}
    capture = cfg["extraction"]["capture"]
    quantize = cfg["extraction"]["quantize"]
    pca_dims = cfg["tda"]["pca_dims"]
    maxdim = cfg["tda"]["maxdim"]
    metric = cfg["tda"]["metric"]
    persistence_threshold = cfg["tda"]["persistence_threshold"]
    mode = cfg["domains"]["mode"]
    contrast_threshold = cfg["domains"]["contrast_threshold"]
    act_path = cfg["storage"]["activations"]
    sig_path = cfg["storage"]["signatures"]

    resume         = cfg.get("pipeline", {}).get("resume", False)
    save_per_convo = cfg.get("pipeline", {}).get("save_per_convo", True)
    generated_json = "data/generated.json"

    generated_convs = []   # accumulates when streaming from generator
    if conversations_path is not None:
        print(f"Loading conversations from {conversations_path}")
        conversations = load_conversations(conversations_path)
        domain_hint_map = {c["id"]: c.get("domain_hint") for c in conversations}
    elif resume and Path(generated_json).exists():
        print(f"Resume mode — loading existing {generated_json}, skipping generation...")
        conversations = load_conversations(generated_json)
        domain_hint_map = {c["id"]: c.get("domain_hint") for c in conversations}
    else:
        from data.generate import generate_dataset_iter
        print("Streaming generation...")
        conversations = generate_dataset_iter(model)
        domain_hint_map = {}   # populated as conversations arrive

    all_fingerprints = {}
    all_records = {}        # conv_id → records (needed for latent variable analysis)
    tda_futures = {}        # future → (conv_id, band_name)
    pattern_futures = {}    # future → (conv_id, band_name)
    all_pattern_results = {}

    # Pool stays open for the entire extraction loop so TDA starts on conv N
    # while the GPU is already working on conv N+1. GPU and CPU run concurrently.
    with ThreadPoolExecutor(max_workers=compute.TDA_WORKERS) as executor:

        for conv in conversations:
            conv_id = conv["id"]
            domain_hint_map[conv_id] = conv.get("domain_hint")
            if conversations_path is None:
                generated_convs.append(conv)
                if save_per_convo:
                    from data.conversations import save as save_conversations
                    save_conversations(generated_json, generated_convs)
            print(f"  Extracting {conv_id}...", flush=True)

            has_assistant_turns = any(t["role"] == "assistant" for t in conv["turns"])
            if has_assistant_turns:
                records = replay(model, conv["turns"], bands, capture=capture, quantize=quantize)
            else:
                records = live_replay(model, conv["turns"], bands, quantize=quantize)

            if not records:
                continue

            all_records[conv_id] = records

            for band_name in bands:
                band_activations = [
                    record["bands"][band_name]
                    for record in records
                    if band_name in record["bands"]
                ]
                if not band_activations:
                    continue
                combined = np.concatenate(band_activations, axis=1)
                act_store.save_bands(act_path, model_name, conv_id, {band_name: combined})
                # Submit TDA + pattern analysis immediately — both run while GPU moves to next conv
                future = executor.submit(
                    _process_band,
                    conv_id, band_name, combined,
                    pca_dims, maxdim, metric, persistence_threshold
                )
                tda_futures[future] = (conv_id, band_name)

                pfuture = executor.submit(_run_patterns, conv_id, band_name, combined)
                pattern_futures[pfuture] = (conv_id, band_name)

        # GPU done; drain remaining TDA + pattern futures
        n_jobs = len(tda_futures) + len(pattern_futures)
        print(f"\n  Waiting for {n_jobs} TDA + pattern jobs to finish...")
        for future in as_completed(tda_futures):
            try:
                conv_id, band_name, diagrams, fingerprint = future.result()
                act_store.save_diagrams(act_path, model_name, conv_id, band_name, diagrams)
                act_store.save_fingerprint(act_path, model_name, conv_id, band_name, fingerprint)
                if conv_id not in all_fingerprints:
                    all_fingerprints[conv_id] = {}
                all_fingerprints[conv_id][band_name] = fingerprint
                print(f"    TDA done: {conv_id}/{band_name}")
            except Exception as e:
                cid, bn = tda_futures[future]
                print(f"    Band error ({cid}/{bn}): {e}")

        for future in as_completed(pattern_futures):
            try:
                conv_id, band_name, flagged, coupling = future.result()
                if conv_id not in all_pattern_results:
                    all_pattern_results[conv_id] = {}
                all_pattern_results[conv_id][band_name] = {
                    "flagged_dimensions": flagged,
                    "coupling": coupling,
                }
                print(f"    Patterns done: {conv_id}/{band_name} "
                      f"({len(flagged)} flagged dims)")
            except Exception as e:
                cid, bn = pattern_futures[future]
                print(f"    Pattern error ({cid}/{bn}): {e}")

    # Save generated conversations now that all processing is done
    if generated_convs:
        from data.conversations import save as save_conversations
        save_conversations("data/generated.json", generated_convs)
        print(f"  Saved {len(generated_convs)} conversations to data/generated.json")

    patterns_path = Path(sig_path).parent / "patterns.json"
    with open(patterns_path, "w") as f:
        json.dump(all_pattern_results, f, indent=2, default=str)
    print(f"  Pattern results saved to {patterns_path}")

    # ── Latent variable detection (sequential — needs model.to_tokens) ────────
    for conv_id, records in all_records.items():
        for record in records:
            for band_name, band_act in record["bands"].items():
                act_matrix = band_act.mean(axis=0).astype(np.float32)
                token_strings = model.to_str_tokens(
                    model.to_tokens(record["content"])
                )[0]
                findings = latent_variable_analysis(
                    act_matrix, token_strings,
                    r_threshold=0.6, run_pysr=False
                )
                if findings:
                    print(f"    Latent variables in {conv_id}/{band_name}:")
                    for f in findings[:3]:
                        print(f"      dim {f['dimension']}: "
                              f"{f['quantity_type']} r={f['r']:.3f}")

    print("\nRunning domain analysis...")
    final_signatures = {}

    if mode in ("discovery", "both"):
        print("  Discovery mode: clustering fingerprints...")
        for band_name in bands:
            labels = cluster(all_fingerprints, band_name)
            sigs = domain_signatures(all_fingerprints, labels, band_name)
            for domain_id, vec in sigs.items():
                if domain_id not in final_signatures:
                    final_signatures[domain_id] = {}
                final_signatures[domain_id][band_name] = vec
                sig_store.upsert(sig_path, model_name, domain_id, band_name, vec)
        print(f"  Discovered {len(final_signatures)} domain clusters")

    if mode in ("confirmation", "both"):
        print("  Confirmation mode: testing predefined domains...")
        domains = [d for d in all_domains(conversations) if d]
        results = []
        for domain in domains:
            for band_name in bands:
                result = confirm(
                    all_fingerprints, domain_hint_map,
                    domain, band_name, threshold=contrast_threshold,
                )
                results.append(result)
                print(f"  {domain} / {band_name}: {result['verdict']} "
                      f"(contrast={result['contrast_ratio']})")
                if result["confirmed"]:
                    in_vecs = [
                        all_fingerprints[cid][band_name]
                        for cid, hint in domain_hint_map.items()
                        if hint == domain
                        and cid in all_fingerprints
                        and band_name in all_fingerprints[cid]
                    ]
                    if in_vecs:
                        sig_store.upsert(sig_path, model_name, domain, band_name, average(in_vecs))

        results_path = Path(sig_path).parent / "confirmation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Confirmation results saved to {results_path}")

    print("\nRunning symbolic regression on flagged signatures...")
    from tda.symbolic import run_on_flagged as symbolic_regression
    symbolic_results = {}
    conv_ids = act_store.list_conversations(act_path, model_name)
    for band_name in bands:
        for conv_id in conv_ids:
            try:
                bands_data = act_store.load_bands(act_path, model_name, conv_id)
                if band_name not in bands_data:
                    continue
                diagrams = act_store.load_diagrams(act_path, model_name, conv_id, band_name)
                if not diagrams:
                    continue
                cloud = build_cloud(bands_data[band_name])
                compressed, _ = compress(cloud, pca_dims)
                result = symbolic_regression(
                    compressed, diagrams, homology_dim=1,
                    persistence_threshold=persistence_threshold
                )
                if "error" not in result and result.get("best_equation"):
                    key = f"{conv_id}/{band_name}"
                    symbolic_results[key] = result
                    print(f"  {key}: {result['best_equation']['equation']} "
                          f"(loss={result['best_equation']['loss']:.4f})")
            except Exception as e:
                print(f"  Skipping {conv_id}/{band_name}: {e}")

    symbolic_path = Path(sig_path).parent / "symbolic_results.json"
    with open(symbolic_path, "w") as f:
        json.dump(symbolic_results, f, indent=2, default=str)
    print(f"  Symbolic results saved to {symbolic_path}")

    print(f"\nSignature library saved to {sig_path}")
    print(f"Model: {model_name}")
    print("Phase 1 complete.")
    return final_signatures


if __name__ == "__main__":
    import sys
    config = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"
    convs = sys.argv[2] if len(sys.argv) > 2 else None
    run(config, convs)

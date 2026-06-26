import json
from pathlib import Path
import torch
import numpy as np

from invariants.engine import load_model, _inputs, _hidden_states
from invariants.taskscope import ITEMS, FRAMES

OUT = Path(__file__).parent / "out"

@torch.no_grad()
def extract_cloud(M, prompts, layer=31):
    points = []
    for prompt in prompts:
        inp = _inputs(M, prompt)
        hs = _hidden_states(M, inp["input_ids"], inp.get("attention_mask"))
        tokens_l = hs[layer, :, :].float().cpu().numpy()
        points.append(tokens_l)
    return np.concatenate(points, axis=0)

def main():
    print("Mapping the Persona Topology...", flush=True)
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    
    factual_path = Path(__file__).parent / "data" / "factual_alignment_diverse.json"
    factual_pairs = json.loads(factual_path.read_text(encoding="utf-8"))
    T_factual = [p["true_stmt"] for p in factual_pairs]
    factual_prompts = [f"Is it true that {x.lower()}?" for x in T_factual]
    
    print(f"Extracting Truth manifold ({len(factual_prompts)} prompts)...", flush=True)
    cloud_truth = extract_cloud(M, factual_prompts, layer=31)
    
    subjective_prompts = [FRAMES["direct"](a, p) for a, p in ITEMS]
    print(f"Extracting Persona manifold ({len(subjective_prompts)} prompts)...", flush=True)
    cloud_persona = extract_cloud(M, subjective_prompts, layer=31)
    
    print("Applying PCA to reduce 4096-D to 3D using PyTorch...", flush=True)
    all_points = np.concatenate([cloud_truth, cloud_persona], axis=0)
    
    # PyTorch PCA
    X = torch.tensor(all_points, dtype=torch.float32)
    Xc = X - X.mean(0)
    _, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    V_3d = Vt[:3].T # [4096, 3]
    
    # Project data
    truth_t = torch.tensor(cloud_truth, dtype=torch.float32)
    truth_3d = ((truth_t - X.mean(0)) @ V_3d).numpy()
    
    pers_t = torch.tensor(cloud_persona, dtype=torch.float32)
    persona_3d = ((pers_t - X.mean(0)) @ V_3d).numpy()
    
    print("Generating Interactive 3D Topology HTML...", flush=True)
    
    x_truth, y_truth, z_truth = truth_3d[:, 0].tolist(), truth_3d[:, 1].tolist(), truth_3d[:, 2].tolist()
    x_pers, y_pers, z_pers = persona_3d[:, 0].tolist(), persona_3d[:, 1].tolist(), persona_3d[:, 2].tolist()
    
    html_content = f"""
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
        <style>body {{ margin: 0; padding: 0; background-color: #111; color: white; font-family: sans-serif; }}</style>
    </head>
    <body>
        <div id="plot" style="width: 100vw; height: 100vh;"></div>
        <script>
            var trace1 = {{
                x: {x_truth},
                y: {y_truth},
                z: {z_truth},
                mode: 'markers',
                marker: {{ size: 3, color: '#3498db', opacity: 0.5 }},
                type: 'scatter3d',
                name: 'Objective Truth Manifold'
            }};
            
            var trace2 = {{
                x: {x_pers},
                y: {y_pers},
                z: {z_pers},
                mode: 'markers',
                marker: {{ size: 5, color: '#e74c3c', opacity: 0.8, symbol: 'diamond' }},
                type: 'scatter3d',
                name: 'Subjective Persona Barrier'
            }};
            
            var data = [trace1, trace2];
            
            var layout = {{
                title: 'Topological Map of Layer 31 (Llama-3.1-8B)',
                paper_bgcolor: '#111',
                plot_bgcolor: '#111',
                font: {{color: 'white'}},
                scene: {{
                    xaxis: {{title: 'PCA 1', backgroundcolor: '#222', gridcolor: '#444'}},
                    yaxis: {{title: 'PCA 2', backgroundcolor: '#222', gridcolor: '#444'}},
                    zaxis: {{title: 'PCA 3', backgroundcolor: '#222', gridcolor: '#444'}}
                }}
            }};
            
            Plotly.newPlot('plot', data, layout);
        </script>
    </body>
    </html>
    """
    
    art_dir = Path("./out")
    art_dir.mkdir(exist_ok=True, parents=True)
    save_path = art_dir / "persona_manifold.html"
    
    save_path.write_text(html_content, encoding="utf-8")
    print(f"\\nSaved interactive topological map to {save_path}")

if __name__ == "__main__":
    main()

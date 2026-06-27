import os
from pathlib import Path
import json

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA

from invariants.engine import load_model, _token_cloud, generate_text

def get_dynamic_bounds(hs, epsilon=0.005):
    """
    hs: [n_layers, d]
    Returns (start, end, velocities) defining the Elastic Scope plateau.
    """
    sim = F.cosine_similarity(hs[:-1], hs[1:], dim=-1)
    velocity = 1.0 - sim
    
    # Identify where the model velocity drops (conceptual stabilization / U-trough)
    in_plateau = velocity < epsilon
    try:
        start = (in_plateau).nonzero(as_tuple=True)[0][0].item()
    except IndexError:
        start = len(hs) // 3
        
    try:
        end = (in_plateau[start:] == False).nonzero(as_tuple=True)[0][0].item() + start
    except IndexError:
        end = len(hs) - 2
        
    return start, end, velocity

def generate_latent_graph(M, prompt, out_dir="invariants/out", label="graph", epsilon=0.005):
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Extracting token cloud for prompt...")
    c = _token_cloud(M, prompt, max_new_tokens=40) # [gen_len, n_layers, d]
    if c is None:
        print("Generation failed to produce enough tokens.")
        return
        
    # Get text
    text = generate_text(M, prompt, max_new_tokens=40)
    
    # Compute bounds on the MEAN trajectory of the generated tokens
    mean_hs = c.mean(dim=0) # [n_layers, d]
    start, end, velocity = get_dynamic_bounds(mean_hs, epsilon)
    
    print(f"Elastic Scope detected plateau bounds: L{start} to L{end}")
    
    # Slice the cloud into three distinct zones
    # 1. Uptake (L0 to start)
    # 2. Plateau (start to end) -> The True Geometry
    # 3. Post-Hoc Translation (end to n_layers) -> The Linguistic Bottleneck
    
    # We will fit a PCA on the entire trajectory to project it down to 2D
    gen_len, n_layers, d = c.shape
    flattened = c.reshape(-1, d).cpu().numpy()
    
    pca = PCA(n_components=2)
    pca.fit(flattened)
    
    # Project each zone
    c_np = c.cpu().numpy()
    
    html = f"""
    <html>
    <head>
        <title>Topological Vector Graph: {label}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>body {{ font-family: sans-serif; margin: 2rem; background: #0d1117; color: #c9d1d9; }}</style>
    </head>
    <body>
        <h2>Elastic Scope: Topological Vector Graph</h2>
        <p><b>Plateau Bounds:</b> Layer {start} - Layer {end}</p>
        <p><b>Variance Threshold:</b> &epsilon; = {epsilon}</p>
        <div style="background: #161b22; padding: 1rem; border-radius: 8px;">
            <p><b>Model Output (The Bottleneck):</b></p>
            <p><i>"{text}"</i></p>
        </div>
        <div id="plot" style="width:100%; height:800px;"></div>
        <script>
    """
    
    traces = []
    
    for t in range(gen_len):
        # Project this token's trajectory
        traj = pca.transform(c_np[t]) # [n_layers, 2]
        
        # Uptake
        traces.append(f"""
        {{
            x: {traj[:start, 0].tolist()},
            y: {traj[:start, 1].tolist()},
            mode: 'lines+markers',
            name: 'Tok {t} Uptake',
            line: {{color: 'rgba(100, 100, 100, 0.3)', width: 1}},
            marker: {{size: 3}},
            showlegend: false
        }}
        """)
        
        # Plateau (True Logic)
        traces.append(f"""
        {{
            x: {traj[start:end, 0].tolist()},
            y: {traj[start:end, 1].tolist()},
            mode: 'lines+markers',
            name: 'Tok {t} Plateau',
            line: {{color: 'rgba(88, 166, 255, 0.8)', width: 3}},
            marker: {{size: 5}},
            showlegend: false
        }}
        """)
        
        # Post-Hoc (Linguistic Bottleneck)
        traces.append(f"""
        {{
            x: {traj[end:, 0].tolist()},
            y: {traj[end:, 1].tolist()},
            mode: 'lines+markers',
            name: 'Tok {t} Post-Hoc',
            line: {{color: 'rgba(248, 81, 73, 0.8)', width: 2}},
            marker: {{size: 4}},
            showlegend: false
        }}
        """)
        
    html += "var data = [" + ",".join(traces) + "];\n"
    html += """
        var layout = {
            plot_bgcolor: '#0d1117',
            paper_bgcolor: '#0d1117',
            font: {color: '#c9d1d9'},
            title: 'Latent Trajectory Divergence (Blue = Plateau, Red = Post-Hoc)',
            xaxis: {showgrid: false, zeroline: false},
            yaxis: {showgrid: false, zeroline: false},
            hovermode: 'closest'
        };
        Plotly.newPlot('plot', data, layout);
        </script>
    </body>
    </html>
    """
    
    out_file = Path(out_dir) / f"{label}_vector_graph.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"Saved Vector Graph to {out_file}")

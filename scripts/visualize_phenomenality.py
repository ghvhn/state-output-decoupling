import json
import os
import webbrowser
import argparse
from pathlib import Path

def get_phenomenality(record: dict) -> dict:
    """Read both historical and current synthesis-record layouts."""
    if not isinstance(record, dict):
        return {}
    top_level = record.get("phenomenality")
    if isinstance(top_level, dict):
        return top_level
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        nested = metadata.get("phenomenality")
        if isinstance(nested, dict):
            return nested
    return {}


def generate_dashboard(json_path: str, output_html: str, open_browser: bool = True):
    print(f"Loading {json_path}...")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    charts_data = []
    
    for row in data.get("rows", []):
        methods = row.get("methods", {})
        if "humble_synthesis" not in methods:
            continue
            
        syn = methods["humble_synthesis"]
        attempts = syn.get("result", {}).get("attempts", [])
        
        # We'll create one chart per item, plotting the phenomenality points across all tokens
        ambiguity = []
        repetition = []
        disagreement = []
        tokens = []
        
        token_counter = 0
        
        for attempt in attempts:
            records = attempt.get("synthesis_records", [])
            for rec in records:
                token_counter += 1
                p = get_phenomenality(rec)
                if p:
                    ambiguity.append(p.get("ambiguity", 0))
                    repetition.append(p.get("repetition", 0))
                    disagreement.append(p.get("disagreement", 0))
                    tokens.append(f"Token {token_counter}")
                    
        if len(tokens) > 0:
            charts_data.append({
                "item_id": row["index"],
                "question": syn.get("result", {}).get("question", "")[:100] + "...",
                "correct": syn.get("correct", False),
                "tokens": tokens,
                "ambiguity": ambiguity,
                "repetition": repetition,
                "disagreement": disagreement
            })
            
    if not charts_data:
        print("No phenomenality data found in the logs! Ensure the benchmark triggered Test-Time Synthesis.")
        return
        
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive Phenomenality Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{ font-family: 'Inter', sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }}
            h1 {{ text-align: center; color: #38bdf8; }}
            .chart-container {{ background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
            .info {{ margin-bottom: 10px; color: #94a3b8; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1>Phenomenality (Cognitive Empathy) Dashboard</h1>
        <div id="charts"></div>
        
        <script>
            const data = {json.dumps(charts_data)};
            const chartsDiv = document.getElementById('charts');
            
            data.forEach((chart, index) => {{
                // Create container
                const div = document.createElement('div');
                div.className = 'chart-container';
                div.innerHTML = `
                    <h3>Item ${{chart.item_id}} (${{chart.correct ? 'Correct \u2714\ufe0f' : 'Failed \u274c'}})</h3>
                    <div class="info">${{chart.question}}</div>
                    <div id="plot-${{index}}"></div>
                `;
                chartsDiv.appendChild(div);
                
                // Plotly trace setup
                const traceAmbiguity = {{
                    x: chart.tokens, y: chart.ambiguity,
                    name: 'Ambiguity', type: 'scatter', line: {{color: '#38bdf8', width: 3}}
                }};
                const traceRepetition = {{
                    x: chart.tokens, y: chart.repetition,
                    name: 'Repetition', type: 'scatter', line: {{color: '#fbbf24', width: 3}}
                }};
                const traceDisagreement = {{
                    x: chart.tokens, y: chart.disagreement,
                    name: 'Disagreement', type: 'scatter', line: {{color: '#f87171', width: 3}}
                }};
                
                const layout = {{
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {{ color: '#f8fafc' }},
                    xaxis: {{ title: 'Generation Timeline', gridcolor: '#334155' }},
                    yaxis: {{ title: 'Phenomenal Score', gridcolor: '#334155', range: [-1, 1] }},
                    margin: {{ t: 20 }}
                }};
                
                Plotly.newPlot(`plot-${{index}}`, [traceAmbiguity, traceRepetition, traceDisagreement], layout);
            }});
        </script>
    </body>
    </html>
    """
    
    html_path = Path(output_html).resolve()
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Dashboard generated at {html_path}")
    if open_browser:
        webbrowser.open_new_tab(f"file://{html_path}")
        print("Launched in browser!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a phenomenality dashboard from a humble benchmark JSON.")
    parser.add_argument("--input", default="invariants/out/humble_full_suite_gsm8k.json")
    parser.add_argument("--output", default="invariants/out/phenomenality_dashboard.html")
    parser.add_argument("--no-open", action="store_true", help="Generate the dashboard without opening a browser tab.")
    args = parser.parse_args()
    generate_dashboard(
        args.input,
        args.output,
        open_browser=not args.no_open,
    )

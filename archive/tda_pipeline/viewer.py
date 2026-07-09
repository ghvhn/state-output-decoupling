import json
from pathlib import Path
import textwrap

def generate_mermaid():
    findings_path = Path("FINDINGS.json")
    if not findings_path.exists():
        print("FINDINGS.json not found.")
        return
    
    with open(findings_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    nodes = data.get("nodes", {})
    edges = data.get("edges", [])
    
    mermaid_lines = ["```mermaid", "graph TD"]
    mermaid_lines.append("    %% Nodes")
    
    for node_id, node_data in nodes.items():
        status = node_data.get("status", "unknown")
        conclusion = node_data.get("conclusion", "")
        # wrap text for mermaid node
        wrapped = "<br>".join(textwrap.wrap(conclusion, width=50))
        
        # Color coding by status using mermaid classes
        shape_start, shape_end = "[", "]"
        if status == "refuted":
            shape_start, shape_end = "((", "))"  # round for refuted
        elif status == "positive":
            shape_start, shape_end = "[/", "\\]" # parallelogram for positive
            
        label = f'"{node_id}<br><b>{status.upper()}</b><br><i>{wrapped}</i>"'
        mermaid_lines.append(f"    {node_id}{shape_start}{label}{shape_end}")
        
    mermaid_lines.append("")
    mermaid_lines.append("    %% Edges")
    for edge in edges:
        source = edge["from"]
        target = edge["to"]
        edge_type = edge["type"]
        
        if edge_type == "informs":
            arrow = "-->"
        elif edge_type == "ruled_out":
            arrow = "-.->|ruled out|"
        elif edge_type == "reframes":
            arrow = "==>|reframes|"
        else:
            arrow = "-->"
            
        mermaid_lines.append(f"    {source} {arrow} {target}")
        
    mermaid_lines.append("")
    mermaid_lines.append("    %% Styling")
    mermaid_lines.append("    classDef positive fill:#d4edda,stroke:#28a745,color:#155724;")
    mermaid_lines.append("    classDef negative fill:#fff3cd,stroke:#ffc107,color:#856404;")
    mermaid_lines.append("    classDef refuted fill:#f8d7da,stroke:#dc3545,color:#721c24;")
    mermaid_lines.append("    classDef method fill:#e2e3e5,stroke:#383d41,color:#383d41;")
    mermaid_lines.append("    classDef mixed fill:#cce5ff,stroke:#007bff,color:#004085;")
    mermaid_lines.append("    classDef in_progress fill:#d1ecf1,stroke:#17a2b8,color:#0c5460;")
    mermaid_lines.append("    classDef planned fill:#f8f9fa,stroke:#6c757d,color:#6c757d,stroke-dasharray: 5 5;")
    
    for node_id, node_data in nodes.items():
        status = node_data.get("status", "unknown")
        if status in ["positive", "negative", "refuted", "method", "mixed", "in_progress", "planned"]:
            mermaid_lines.append(f"    class {node_id} {status};")
            
    mermaid_lines.append("```")
    
    out_path = Path("SPINE_VIEWER.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Project Epistemic Spine\n\n")
        f.write("This diagram is generated from `FINDINGS.json`. It maps the logical argument of the project.\n\n")
        f.write("\n".join(mermaid_lines))
        f.write("\n")
        
    print(f"Generated {out_path}")

if __name__ == '__main__':
    generate_mermaid()

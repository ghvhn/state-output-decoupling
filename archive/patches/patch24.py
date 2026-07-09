import os

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''def build_probe_init_macro(probes):
    """Regenerate the CURRENT probe set as replayable commands, derived from each
    probe's stored origin (its framings field): a minted probe re-mints from its
    contrastive text, an adopted probe re-adopts its source dimension, a composed
    probe re-composes its recipe. Exposed probes get a trailing :probe expose.
    Reconstructs the sensors from text -- no binary weights needed to share."""
    lines = ["# Regenerates this session's probes. Replay with :run self (or its alias)."]
    for name in sorted(probes):
        fr = probes[name].get("framings") or ("", "")
        a = fr[0] if len(fr) > 0 else ""
        b = fr[1] if len(fr) > 1 else ""
        if a.startswith("adopted:"):
            lines.append(f":probe adopt {name}")
        elif a.startswith("composed:"):
            recipe = a.split(":", 1)[1].strip()
            lines.append(f":probe compose {name} {recipe}")
        elif a or b:
            lines.append(f":probe {name} {a} || {b}")
        else:
            lines.append(f"# {name}: no framings stored -- cannot regenerate from text (only its .pt weights hold it)")
            continue
        if probes[name].get("exposed"):
            lines.append(f":probe expose {name}")
    return lines'''

replacement = '''def build_probe_init_macro(probes):
    """Regenerate the CURRENT probe set as replayable commands, derived from each
    probe's stored origin (its framings field): a minted probe re-mints from its
    contrastive text, an adopted probe re-adopts its source dimension, a composed
    probe re-composes its recipe. Exposed probes get a trailing :probe expose.
    Reconstructs the sensors from text -- no binary weights needed to share."""
    lines = ["# Regenerates this session's probes. Replay with :run self (or its alias)."]
    
    deps = {}
    for name in probes:
        deps[name] = set()
        fr = probes[name].get("framings") or ("", "")
        a = fr[0] if len(fr) > 0 else ""
        if a.startswith("composed:"):
            recipe = a.split(":", 1)[1].strip()
            terms, _ = parse_compose_expr(recipe)
            for _, tname in terms:
                if tname in probes:
                    deps[name].add(tname)
                    
    sorted_probes = []
    visited = set()
    temp_mark = set()
    
    def visit(n):
        if n in temp_mark:
            return
        if n not in visited:
            temp_mark.add(n)
            for m in sorted(deps.get(n, set())):
                visit(m)
            temp_mark.remove(n)
            visited.add(n)
            sorted_probes.append(n)
            
    for name in sorted(probes.keys()):
        if name not in visited:
            visit(name)
            
    for name in sorted_probes:
        fr = probes[name].get("framings") or ("", "")
        a = fr[0] if len(fr) > 0 else ""
        b = fr[1] if len(fr) > 1 else ""
        if a.startswith("adopted:"):
            lines.append(f":probe adopt {name}")
        elif a.startswith("composed:"):
            recipe = a.split(":", 1)[1].strip()
            lines.append(f":probe compose {name} {recipe}")
        elif a or b:
            lines.append(f":probe {name} {a} || {b}")
        else:
            lines.append(f"# {name}: no framings stored -- cannot regenerate from text (only its .pt weights hold it)")
            continue
        if probes[name].get("exposed"):
            lines.append(f":probe expose {name}")
    return lines'''

if target in content:
    content = content.replace(target, replacement)
    print("Patched topological sort")
else:
    print("Failed to patch topological sort")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

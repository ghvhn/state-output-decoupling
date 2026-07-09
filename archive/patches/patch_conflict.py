import re

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''<<<<<<< HEAD
        if v.shape[0] == model.d_model:
            n = v.norm()
            if n.item() > 0:
                unit = v / n
                n_layers = int(model.model.config.num_hidden_layers)
                targets = sorted(want) if want is not None else steer_band_layers(n_layers)
                for L in targets:
                    if 0 <= int(L) < n_layers:
                        direction[int(L)] = unit.clone()
    return direction, os.path.basename(src_path)
=======
        n = v.norm()
        if n.item() > 0:
            unit = v / n
            n_layers = int(model.model.config.num_hidden_layers)
            targets = sorted(want) if want is not None else steer_band_layers(n_layers)
            for L in targets:
                if 0 <= int(L) < n_layers:
                    direction[int(L)] = unit.clone()
    return direction, os.path.basename(src_path), exposed
>>>>>>> origin/claude/cpu-only-feature-parity-vdpjzk'''

replacement = '''        if v.shape[0] == model.d_model:
            n = v.norm()
            if n.item() > 0:
                unit = v / n
                n_layers = int(model.model.config.num_hidden_layers)
                targets = sorted(want) if want is not None else steer_band_layers(n_layers)
                for L in targets:
                    if 0 <= int(L) < n_layers:
                        direction[int(L)] = unit.clone()
    return direction, os.path.basename(src_path), exposed'''

if target in content:
    content = content.replace(target, replacement)
    print("Resolved conflict")
else:
    print("Conflict not found")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

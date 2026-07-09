import re

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''                    # otherwise margs[1] is the knob to match to
                    knob = margs[1]
                    if knob not in knobs and f"probe_{knob}" not in tuner.triggers and knob not in ("steer_cap_fraction", "steer_band"):'''

replacement = '''                    # otherwise margs[1] is the knob to match to
                    knob = margs[1]
                    if knob.upper() in ("CHOICE", "CHOOSE", "AUTO"):
                        valid_knobs = list(knobs) + ["steer_cap_fraction", "steer_band"]
                        resolved_knob = resolve_probe_choice(knob, valid_knobs, model=model, config=config, action_name="match_target")
                        if not resolved_knob:
                            continue
                        knob = resolved_knob
                    if knob not in knobs and f"probe_{knob}" not in tuner.triggers and knob not in ("steer_cap_fraction", "steer_band"):'''

if target in content:
    content = content.replace(target, replacement)
    print("Patched :probe match choose logic")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

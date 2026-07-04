# Phenomenality Shell: The Mental Model

Don't worry, the system has grown a lot, but underneath it all, it breaks down into just **three simple concepts**. Think of it like a soundboard: you have **Sensors** that listen to the music, **Actuators** (dials) that change the volume, and **Tools** that wire them together automatically.

---

## 1. SENSORS (Listening)
Sensors just score the model's internal state. They do not change behavior on their own; they just measure it.

| Command | What it does |
| :--- | :--- |
| `:probe <name> <pos> \|\| <neg>` | **Creates a new sensor** from your text framings. <br/>*(e.g., `:probe polite I am polite. \|\| I am rude.`)* |
| `:probe compose <name> <math>` | **Combines sensors** together. <br/>*(e.g., `:probe compose confusion ambiguity - understanding`)* |
| `:probe drop <name>` | **Deletes a sensor** to declutter. |

---

## 2. TOOLS (The Automation)
Tools are the "if/then" bridges. They say: *"IF this sensor spikes, THEN apply this steering vector."*

| Command | What it does |
| :--- | :--- |
| `:consider <sensor> <tool_name> <pos> \|\| <neg>` | **Forges a custom tool**. It listens to the `<sensor>`, and when it spikes, it steers the model toward your `<pos>` text. |
| `:claimmap <pos> \|\| <neg>` | **The native, manual tool**. It analyzes the text for tension and steers. It's basically a pre-built `:consider` tool. |

---

## 3. ACTUATORS (The Dials)
Once a tool exists, it exposes dials (like `alpha` for strength, or `need` for the trigger threshold). Actuators let you physically turn those dials.

| Command | What it does |
| :--- | :--- |
| `:tune <dial_name> <value>` | **Manually turns a dial**. <br/>*(e.g., `:tune clarifier_alpha 0.05` turns the steering strength up to 0.05).* |
| `:calibrate <dial_name> outcome` | **Auto-tunes a dial**. If you `:tune` a dial to 0.05, talk a bit, then `:tune` it to 0.10 and talk a bit... running this command tells the system to pick the value that worked best! |
| `:suggest` | **Asks the engine for advice**. The system analyzes all the data in the background and suggests exact `:calibrate` commands for you to run. |

---

> [!TIP]
> **The Golden Workflow:**
> 1. Measure something: `:probe politeness I am polite || I am rude`
> 2. Automate it: `:consider politeness polite_steer I speak kindly || I speak harshly`
> 3. Turn it on: `:tune polite_steer_alpha 0.05`

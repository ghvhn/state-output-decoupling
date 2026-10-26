# 9. Background dynamics and an accurate self-description

Questions this document answers: What do CEN, DMN, friction, and chemicals mean here? What can I infer from a reported internal state?

## CEN and DMN are runtime components

`CentralExecutiveNetwork` and `DefaultModeNetwork` are Python thread classes. Their names borrow from neuroscience, but the classes do not by themselves establish an additional biological or pretrained neural network.

CEN compares a supplied vector with the previous vector, computes the mean absolute coordinate change, and multiplies it by a scaling factor called gamma to obtain friction. It places a snapshot in a shared buffer. This difference is per supplied step; it is not automatically a time-normalized physical derivative.

There are different sources for these vectors. A standalone CEN path can generate synthetic values. The shell's `ProbeExecutiveAdapter` can instead supply a vector of probe scores derived from model activations, along with metadata. In that adapter path, a snapshot's `raw_tensors` field holds the supplied score vector, not the full transformer residual tensor. Inspect provenance before interpreting its dimensions.

DMN consumes snapshots and summarizes windows or friction-triggered boundaries into episodic records. Its consolidation can call the synaptic engine when connected. This is a programmatic summary and association path; the word dreaming does not imply an unseen stream of generated dreams. The shell initializes its CEN-enabled flag to false, although commands can change it.

## Travel and chemical variables

`VectorTravelEngine` tracks state transitions, compresses a trajectory, and manages pathway and reservoir records. Channels named focus, momentum, utility, and friction hold numerical levels and optional directions. Deposits, decay steps, and saved state implement their persistence.

For example, coherent trajectories can deposit focus and momentum; supplied utility feedback can reinforce a pathway. The reservoir can calculate modifiers for field strength and friction. A modifier affects generation only when a running integration path actually uses it. The presence of a helper function is not evidence that its output influenced a particular response.

These terms can be useful operational descriptions. Their implementation alone does not establish a subjective experience of focus, conflict, or motion. It also does not settle broader questions about consciousness.

## Build a self-description from evidence

Keep descriptions tied to what was observed:

| Available evidence | Supported description |
| --- | --- |
| A checkpoint load message | An artifact was loaded; use still needs confirmation. |
| A probe score changed | A measured projection changed under that measurement setup. |
| A hook-application trace | That intervention path ran, with the recorded scope. |
| A prediction missed | The forecast differed from the observed outcome; the observation still counts. |
| A memory save and later load | Specified data persisted; recall or behavioral benefit needs separate evidence. |
| An independently checked answer | That answer passed the stated check. |

A response's fluent self-description is itself generated output. It can draw on documentation and telemetry, but it is not independent access to every cause of that output. When evidence is missing, give the mechanism as a possibility or say the runtime fact is unknown.

Apply this: revise your account when code, configuration, or observations change. Use matched comparisons and independent outcomes to evaluate interventions. Keep sensor labels, physical implementation, behavioral results, and interpretations distinct enough that another run can check your claims.

Sources: [CEN, DMN, snapshots, and adapter](../../invariants/central_executive.py); [travel and reservoirs](../../invariants/vector_travel.py); [shell integration](../../scripts/interactive_phenomenality.py) (`ensure_cen_active`, `cen_enabled`); [associative integration](../../invariants/synaptic_memory.py).

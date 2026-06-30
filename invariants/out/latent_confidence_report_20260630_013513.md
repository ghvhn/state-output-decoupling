# Latent Confidence Benchmark (activations decide; English only scored)

- 24 problems, k=6

- single-pass accuracy:        62%
- majority vote (English):     71%

## Reading ONLY activations to gate commitment:
- accuracy on SETTLED-framing third (low disp): **88%**
- accuracy on SCATTERED-framing third (high disp): 62%
- activation-dispersion vs correctness: r=-0.109, perm-p=0.6226

## Selective curve (answer lowest-dispersion fraction, abstain on rest):
  answer lowest-disp 33% (n=8): accuracy 88%
  answer lowest-disp 50% (n=12): accuracy 83%
  answer lowest-disp 66% (n=16): accuracy 75%
  answer lowest-disp 100% (n=24): accuracy 71%

WIN if settled-third accuracy >> scattered-third: the latent alone knows which of the model's answers to trust. Then confident-answer-the-settled + abstain-the-rest beats flat accuracy on calibration -- the egg, realized, from activations only.
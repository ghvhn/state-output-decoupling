import subprocess
import time


def main():
    print("Starting CoT perturbation probe...")
    cmd = ["python", "-u", "-m", "invariants.cot_perturb"]
    print(f"\nRunning: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"\nCoT perturbation probe complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

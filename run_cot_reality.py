import subprocess
import time


def main():
    print("Starting CoT reality probe...")
    cmd = ["python", "-u", "-m", "invariants.cot_reality"]
    print(f"\nRunning: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"\nCoT reality probe complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

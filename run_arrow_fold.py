import subprocess
import time


def main():
    print("Starting arrow/fold probe...")
    cmd = ["python", "-u", "-m", "invariants.arrow_fold"]
    print(f"\nRunning: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"\nArrow/fold probe complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

import subprocess
import time


def main():
    print("Starting translation/thinking v2 probe...")
    cmd = ["python", "-u", "-m", "invariants.translation_thinking_v2"]
    print(f"\nRunning: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"\nTranslation/thinking v2 probe complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

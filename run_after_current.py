import subprocess
import time


def main():
    print("Starting queued follow-up batch...")
    scripts = [
        ["python", "-u", "-m", "invariants.intent_surface_control"],
    ]
    for cmd in scripts:
        print(f"\nRunning: {' '.join(cmd)}")
        t0 = time.time()
        subprocess.run(cmd, check=True)
        print(f"Finished in {time.time() - t0:.1f}s")
    print("\nQueued follow-up batch complete!")


if __name__ == "__main__":
    main()

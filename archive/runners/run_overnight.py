import subprocess
import time

def main():
    print("Starting Final Batch...")
    scripts = [
        ["python", "-u", "-m", "invariants.reflexive_decompose"],
        ["python", "-u", "-m", "invariants.reflexive", "--n", "30"]
    ]
    for cmd in scripts:
        print(f"\nRunning: {' '.join(cmd)}")
        t0 = time.time()
        subprocess.run(cmd, check=True)
        print(f"Finished in {time.time() - t0:.1f}s")
        
    print("\nFinal Batch Complete!")

if __name__ == "__main__":
    main()

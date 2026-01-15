import sys
import pandas as pd
import matplotlib.pyplot as plt

def main(csv_path: str):
    df = pd.read_csv(csv_path)

    # Ensure numeric where needed
    df["t_sec"] = pd.to_numeric(df["t_sec"], errors="coerce")
    df["run"] = pd.to_numeric(df["run"], errors="coerce")

    ir_cols = [f"ir{i}" for i in range(8)]

    runs = sorted(df["run"].dropna().unique())

    for r in runs:
        sub = df[df["run"] == r].copy()

        plt.figure()
        for c in ir_cols:
            plt.plot(sub["t_sec"], pd.to_numeric(sub[c], errors="coerce"), label=c)

        plt.title(f"Task 0 – IR sensors over time (run {int(r)})")
        plt.xlabel("Time (s)")
        plt.ylabel("IR value")
        plt.legend()
        plt.show()

    # Optional: plot front_signal + thresholds (useful for report)
    if "front_signal" in df.columns:
        for r in runs:
            sub = df[df["run"] == r].copy()
            plt.figure()
            plt.plot(sub["t_sec"], pd.to_numeric(sub["front_signal"], errors="coerce"), label="front_signal")
            plt.plot(sub["t_sec"], pd.to_numeric(sub["near_th"], errors="coerce"), label="near_th")
            plt.plot(sub["t_sec"], pd.to_numeric(sub["clear_th"], errors="coerce"), label="clear_th")
            plt.title(f"Task 0 – front signal vs thresholds (run {int(r)})")
            plt.xlabel("Time (s)")
            plt.ylabel("Value")
            plt.legend()
            plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_task0_ir.py results/task0_ir_XXXX.csv")
        sys.exit(1)
    main(sys.argv[1])

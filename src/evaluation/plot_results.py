"""
plot_results.py — Phase 4 performance visualisation
Generates phase4_performance_graph.png in the project root.
Run from the project root: python src/evaluation/plot_results.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Path (relative to project root, works on any OS) ──────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH  = os.path.join(BASE_DIR, "data", "results", "minsky_phase4_results.csv")
PLOT_PATH = os.path.join(BASE_DIR, "phase4_performance_graph.png")

if not os.path.exists(CSV_PATH):
    print(f"CSV not found at {CSV_PATH}. Run run_minsky.py first.")
    raise SystemExit(1)

df = pd.read_csv(CSV_PATH)

# ── Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Minsky Motor Abstractions — Phase 4 Validation", fontsize=14, fontweight="bold")

# Left: reward per trial
ax1 = axes[0]
ax1.plot(df["Trial"], df["Total_Reward"], marker="o", markersize=4,
         color="#1F4E79", linewidth=1.5, label="Cumulative Reward")
ax1.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax1.set_title("Reward Stability Across 50 Trials", fontsize=12)
ax1.set_xlabel("Trial Number", fontsize=11)
ax1.set_ylabel("Total Cumulative Reward", fontsize=11)
ax1.set_ylim(df["Total_Reward"].min() - 20, df["Total_Reward"].max() + 20)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(fontsize=10)

# Right: rolling success rate
ax2 = axes[1]
if "Status_Code" in df.columns:
    df["RollingSuccessRate"] = df["Status_Code"].expanding().mean() * 100
    ax2.plot(df["Trial"], df["RollingSuccessRate"], marker="s", markersize=4,
             color="#2E74B5", linewidth=1.5, label="Cumulative Success Rate")
    ax2.set_ylim(0, 110)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax2.axhline(y=100, color="green", linewidth=0.8, linestyle="--", alpha=0.6, label="100% target")
    ax2.set_title("Cumulative Success Rate", fontsize=12)
    ax2.set_xlabel("Trial Number", fontsize=11)
    ax2.set_ylabel("Success Rate (%)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
print(f"Graph saved to: {PLOT_PATH}")
plt.show()

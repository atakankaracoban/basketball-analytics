from nba_api.stats.endpoints import leaguedashteamstats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import time

print("Fetching team stats...")

general = leaguedashteamstats.LeagueDashTeamStats(
    season="2025-26",
    measure_type_detailed_defense="Base"
)
time.sleep(1)

df = general.get_data_frames()[0]

# Per game stats
df["PPG"]     = (df["PTS"] / df["GP"]).round(2)
df["RPG"]     = (df["REB"] / df["GP"]).round(2)
df["APG"]     = (df["AST"] / df["GP"]).round(2)
df["TOPG"]    = (df["TOV"] / df["GP"]).round(2)
df["SPG"]     = (df["STL"] / df["GP"]).round(2)
df["BPG"]     = (df["BLK"] / df["GP"]).round(2)
df["FG_PCT"]  = (df["FGM"] / df["FGA"]).round(3)
df["FG3_PCT"] = (df["FG3M"] / df["FG3A"]).round(3)
df["WIN_PCT"] = df["W_PCT"].round(3)

stats_to_test = {
    "Points/Game":     "PPG",
    "Rebounds/Game":   "RPG",
    "Assists/Game":    "APG",
    "Turnovers/Game":  "TOPG",
    "Steals/Game":     "SPG",
    "Blocks/Game":     "BPG",
    "FG%":             "FG_PCT",
    "3PT%":            "FG3_PCT",
}

results = []

print("\n=== CORRELATION WITH WINNING (2025-26 NBA Season) ===\n")
print(f"{'Stat':<25} {'r value':>10} {'p-value':>12} {'Strength':>15}")
print("─" * 65)

for stat_name, col in stats_to_test.items():
    if col in df.columns:
        clean = df[[col, "WIN_PCT"]].dropna()
        r, p = stats.pearsonr(clean[col], clean["WIN_PCT"])

        abs_r = abs(r)
        if abs_r >= 0.7:
            strength = "STRONG"
        elif abs_r >= 0.4:
            strength = "MODERATE"
        elif abs_r >= 0.2:
            strength = "WEAK"
        else:
            strength = "NEGLIGIBLE"

        direction = "+" if r > 0 else "-"
        results.append({
            "Stat": stat_name,
            "r": round(r, 3),
            "p": round(p, 4),
            "Strength": strength,
            "col": col
        })

        print(f"{stat_name:<25} {direction}{abs(r):.3f}{'':>6} "
              f"{p:.4f}{'':>6} {strength}")

results_df = pd.DataFrame(results)
results_df["abs_r"] = results_df["r"].abs()
results_df = results_df.sort_values("abs_r", ascending=True)

# --- VISUALIZATION ---
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.patch.set_facecolor("#1a1a2e")
fig.suptitle("Which Stats Actually Predict Winning?\n2025-26 NBA Season Correlation Analysis",
             fontsize=14, color="white", fontweight="bold")

axes_flat = axes.flatten()

for idx, row in enumerate(results_df.itertuples()):
    if idx >= 8:
        break

    ax = axes_flat[idx]
    ax.set_facecolor("#16213e")

    col = row.col
    clean = df[[col, "WIN_PCT", "TEAM_NAME"]].dropna()

    scatter_colors = plt.cm.RdYlGn(
        (clean["WIN_PCT"] - clean["WIN_PCT"].min()) /
        (clean["WIN_PCT"].max() - clean["WIN_PCT"].min())
    )

    ax.scatter(clean[col], clean["WIN_PCT"],
               c=scatter_colors, s=60, alpha=0.8, zorder=3)

    z = np.polyfit(clean[col], clean["WIN_PCT"], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(clean[col].min(), clean[col].max(), 100)
    ax.plot(x_line, p_line(x_line), "--",
            color="yellow", alpha=0.7, linewidth=1.5)

    for _, team_row in clean.iterrows():
       if (team_row["WIN_PCT"] > clean["WIN_PCT"].quantile(0.85) or
            team_row["WIN_PCT"] < clean["WIN_PCT"].quantile(0.15) or
            "Knicks" in team_row["TEAM_NAME"]):
        ax.annotate(team_row["TEAM_NAME"].split()[-1],
                        (team_row[col], team_row["WIN_PCT"]),
                        fontsize=6, color="white", alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")

    r_val = row.r
    r_color = "lightgreen" if abs(r_val) >= 0.4 else "salmon"

    ax.set_title(f"{row.Stat}\nr = {r_val:+.3f} ({row.Strength})",
                 color=r_color, fontsize=9)
    ax.set_xlabel(row.Stat, color="white", fontsize=8)
    ax.set_ylabel("Win %", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    ax.grid(color="gray", alpha=0.2)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333366")

plt.tight_layout()
plt.savefig("correlation.png", dpi=150, bbox_inches="tight",
            facecolor="#1a1a2e")
print("\nSaved as correlation.png")
plt.show()
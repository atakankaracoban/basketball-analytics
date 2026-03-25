from nba_api.stats.endpoints import leaguedashplayerstats, playercareerstats
from nba_api.stats.static import players
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats
import time

print("Building Draft Prospect Evaluator...")
print("Fetching current season data...\n")

# ── Fetch current and last season data ─────────────────────
def get_season_stats(season):
    time.sleep(0.5)
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame"
    ).get_data_frames()[0]
    return df[df["GP"] >= 20].copy()

df_current = get_season_stats("2025-26")
df_last = get_season_stats("2024-25")

# ── Filter young players ────────────────────────────────────
# NBA experience: rookies = 0, second year = 1
# We use player age as proxy since API doesn't always have experience
df_current = df_current[df_current["AGE"] <= 23].copy()
df_current = df_current[df_current["MIN"] >= 15].copy()  # Meaningful minutes

print(f"Found {len(df_current)} young players (age ≤ 23, ≥15 min/game)\n")

# ── Calculate Prospect Score ────────────────────────────────
# Age adjustment factor — younger players get a bonus
# A 19-year-old doing what a 23-year-old does is more impressive
def age_factor(age):
    """Younger = higher multiplier. 19yo gets 1.4x, 23yo gets 1.0x"""
    return max(1.0, 1.0 + (23 - age) * 0.1)

df_current["AGE_FACTOR"] = df_current["AGE"].apply(age_factor)

# Base weighted efficiency (our formula)
df_current["BASE_SCORE"] = (
    (df_current["PTS"]  * 1.0) +
    (df_current["REB"]  * 1.2) +
    (df_current["AST"]  * 1.5) +
    (df_current["STL"]  * 2.0) +
    (df_current["BLK"]  * 1.3) -
    (df_current["TOV"]  * 2.0)
) * df_current["FG_PCT"]

# Age-adjusted prospect score
df_current["PROSPECT_SCORE"] = (
    df_current["BASE_SCORE"] * df_current["AGE_FACTOR"]
).round(2)

# ── Year-over-year improvement ──────────────────────────────
# Check if player appeared last season too
df_last["BASE_SCORE_LAST"] = (
    (df_last["PTS"]  * 1.0) +
    (df_last["REB"]  * 1.2) +
    (df_last["AST"]  * 1.5) +
    (df_last["STL"]  * 2.0) +
    (df_last["BLK"]  * 1.3) -
    (df_last["TOV"]  * 2.0)
) * df_last["FG_PCT"]

# Merge with last season
df_merged = df_current.merge(
    df_last[["PLAYER_ID", "BASE_SCORE_LAST"]],
    on="PLAYER_ID",
    how="left"
)

# Calculate improvement
df_merged["IMPROVEMENT"] = (
    df_merged["BASE_SCORE"] - df_merged["BASE_SCORE_LAST"]
).round(2)

# For true rookies (no last season data), improvement = 0
df_merged["IMPROVEMENT"] = df_merged["IMPROVEMENT"].fillna(0)
df_merged["IS_ROOKIE"] = df_merged["BASE_SCORE_LAST"].isna()

# ── Final prospect ranking ──────────────────────────────────
# Combine prospect score with improvement trajectory
df_merged["FINAL_SCORE"] = (
    df_merged["PROSPECT_SCORE"] * 0.7 +  # Current production
    df_merged["IMPROVEMENT"] * 0.3        # Trajectory
).round(2)

df_merged = df_merged.sort_values("FINAL_SCORE", ascending=False)
top20 = df_merged.head(20).reset_index(drop=True)

# Print rankings
print("=" * 70)
print(f"{'RANK':<5} {'PLAYER':<25} {'AGE':<5} {'SCORE':<8} "
      f"{'IMPROVE':<10} {'STATUS'}")
print("=" * 70)
for i, row in top20.iterrows():
    status = "ROOKIE" if row["IS_ROOKIE"] else f"+{row['IMPROVEMENT']:.1f}"
    print(f"{i+1:<5} {row['PLAYER_NAME']:<25} {int(row['AGE']):<5} "
          f"{row['FINAL_SCORE']:<8} {status:<10} "
          f"{'⭐' if row['FINAL_SCORE'] > 15 else ''}")

# ── Visualization ───────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("#0d1117")
gs_layout = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle("NBA Draft Prospect Evaluator — 2025-26 Season\n"
             "Age ≤ 23 | Min 15 minutes/game | Age-Adjusted Scoring",
             fontsize=15, fontweight="bold", color="white", y=0.98)

# ── Panel 1: Top 20 Prospect Rankings ──────────────────────
ax1 = fig.add_subplot(gs_layout[0, :])
ax1.set_facecolor("#161b22")

colors = []
for i, row in top20.iterrows():
    if row["FINAL_SCORE"] > 20: colors.append("gold")
    elif row["FINAL_SCORE"] > 12: colors.append("steelblue")
    else: colors.append("cadetblue")

bars = ax1.barh(
    [f"{row['PLAYER_NAME']} ({int(row['AGE'])})"
     for _, row in top20.iterrows()],
    top20["FINAL_SCORE"],
    color=colors, alpha=0.85
)

# Add value labels
for bar, (_, row) in zip(bars, top20.iterrows()):
    label = f"{row['FINAL_SCORE']}"
    if not row["IS_ROOKIE"] and row["IMPROVEMENT"] > 0:
        label += f"  ↑{row['IMPROVEMENT']:.1f}"
    elif not row["IS_ROOKIE"] and row["IMPROVEMENT"] < 0:
        label += f"  ↓{abs(row['IMPROVEMENT']):.1f}"
    else:
        label += "  (R)"
    ax1.text(bar.get_width() + 0.1,
             bar.get_y() + bar.get_height()/2,
             label, va="center", color="white", fontsize=8)

ax1.invert_yaxis()
ax1.set_xlabel("Prospect Score (Age-Adjusted)", color="white", fontsize=10)
ax1.set_title("Top 20 Prospects — Final Rankings\n"
              "Gold = Elite | Blue = Strong | (R) = Rookie | ↑↓ = YoY change",
              color="white", fontsize=11)
ax1.tick_params(colors="white", labelsize=9)
ax1.set_xlim(0, top20["FINAL_SCORE"].max() * 1.2)
for spine in ax1.spines.values():
    spine.set_edgecolor("#333366")

# ── Panel 2: Age vs Score scatter ──────────────────────────
ax2 = fig.add_subplot(gs_layout[1, 0])
ax2.set_facecolor("#161b22")

scatter_colors = plt.cm.RdYlGn(
    (df_merged["FINAL_SCORE"] - df_merged["FINAL_SCORE"].min()) /
    (df_merged["FINAL_SCORE"].max() - df_merged["FINAL_SCORE"].min())
)
ax2.scatter(df_merged["AGE"], df_merged["FINAL_SCORE"],
            c=scatter_colors, s=60, alpha=0.8)

# Label top 5
for _, row in top20.head(5).iterrows():
    ax2.annotate(row["PLAYER_NAME"].split()[-1],
                 (row["AGE"], row["FINAL_SCORE"]),
                 color="white", fontsize=8,
                 xytext=(3, 3), textcoords="offset points")

# Trend line
z = np.polyfit(df_merged["AGE"], df_merged["FINAL_SCORE"], 1)
p = np.poly1d(z)
x_line = np.linspace(df_merged["AGE"].min(), df_merged["AGE"].max(), 100)
ax2.plot(x_line, p(x_line), "--", color="yellow", alpha=0.6, linewidth=1.5)

ax2.set_xlabel("Age", color="white", fontsize=10)
ax2.set_ylabel("Prospect Score", color="white", fontsize=10)
ax2.set_title("Age vs Prospect Score\n(Younger + Better = Elite Prospect)",
              color="white", fontsize=11)
ax2.tick_params(colors="white")
ax2.grid(color="gray", alpha=0.2)
for spine in ax2.spines.values():
    spine.set_edgecolor("#333366")

# ── Panel 3: Improvement scatter ───────────────────────────
ax3 = fig.add_subplot(gs_layout[1, 1])
ax3.set_facecolor("#161b22")

# Only second year players have improvement data
df_second = df_merged[~df_merged["IS_ROOKIE"]].copy()

imp_colors = plt.cm.RdYlGn(
    (df_second["IMPROVEMENT"] - df_second["IMPROVEMENT"].min()) /
    (df_second["IMPROVEMENT"].max() - df_second["IMPROVEMENT"].min() + 0.001)
)
ax3.scatter(df_second["BASE_SCORE_LAST"], df_second["BASE_SCORE"],
            c=imp_colors, s=60, alpha=0.8)

# Diagonal line — points above = improved, below = declined
max_val = max(df_second["BASE_SCORE"].max(),
              df_second["BASE_SCORE_LAST"].max())
ax3.plot([0, max_val], [0, max_val], "--",
         color="gray", alpha=0.5, linewidth=1.5, label="No change")

# Label top improvers
top_improvers = df_second.nlargest(5, "IMPROVEMENT")
for _, row in top_improvers.iterrows():
    ax3.annotate(row["PLAYER_NAME"].split()[-1],
                 (row["BASE_SCORE_LAST"], row["BASE_SCORE"]),
                 color="white", fontsize=8,
                 xytext=(3, 3), textcoords="offset points")

ax3.set_xlabel("Last Season Score", color="white", fontsize=10)
ax3.set_ylabel("This Season Score", color="white", fontsize=10)
ax3.set_title("Year-over-Year Development\n"
              "Above diagonal = improved, Below = declined",
              color="white", fontsize=11)
ax3.legend(facecolor="#161b22", labelcolor="white", fontsize=9)
ax3.tick_params(colors="white")
ax3.grid(color="gray", alpha=0.2)
for spine in ax3.spines.values():
    spine.set_edgecolor("#333366")

plt.savefig("prospect_evaluator.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
print("\nSaved as prospect_evaluator.png")
plt.show()
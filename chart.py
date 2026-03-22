from nba_api.stats.endpoints import leagueleaders
import pandas as pd
import matplotlib.pyplot as plt

print("Fetching data...")

leaders = leagueleaders.LeagueLeaders(
    season="2025-26",
    stat_category_abbreviation="PTS"
)

df = leaders.get_data_frames()[0]

# Include turnovers this time
df = df[["PLAYER", "TEAM", "GP", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG_PCT"]]

# --- WEIGHTED METRIC ---
# Each stat multiplied by its research-based weight
df["Weighted_Score"] = (
    (df["PTS"]  * 1.0) +
    (df["REB"]  * 1.2) +
    (df["AST"]  * 1.5) +
    (df["STL"]  * 2.0) +
    (df["BLK"]  * 1.3) -
    (df["TOV"]  * 2.0)   # Turnovers HURT
) * df["FG_PCT"]

df["Weighted_Score"] = df["Weighted_Score"].round(2)
df = df.sort_values("Weighted_Score", ascending=False).head(20)

# --- CHART ---
plt.figure(figsize=(13, 9))

colors = ["gold" if i == 0 else "steelblue" for i in range(len(df))]
bars = plt.barh(df["PLAYER"], df["Weighted_Score"], color=colors)

for bar, val in zip(bars, df["Weighted_Score"]):
    plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
             f"{val}", va="center", fontsize=9)

plt.xlabel("Weighted Efficiency Score", fontsize=12)
plt.title("Top 20 NBA Players — Weighted Efficiency (2025-26)", fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("weighted_efficiency.png", dpi=150, bbox_inches="tight")
print("Chart saved as weighted_efficiency.png")
plt.show()
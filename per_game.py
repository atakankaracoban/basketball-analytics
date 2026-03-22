from nba_api.stats.endpoints import leagueleaders
import pandas as pd
import matplotlib.pyplot as plt

print("Fetching data...")

leaders = leagueleaders.LeagueLeaders(
    season="2025-26",
    stat_category_abbreviation="PTS"
)

df = leaders.get_data_frames()[0]
df = df[["PLAYER", "TEAM", "GP", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG_PCT"]]

# Filter minimum 30 games — eliminates small sample sizes
df = df[df["GP"] >= 30]

# Convert all stats to PER GAME
for stat in ["PTS", "REB", "AST", "STL", "BLK", "TOV"]:
    df[stat] = (df[stat] / df["GP"]).round(3)

# Weighted formula on per game stats — measures QUALITY
df["Quality_Score"] = (
    (df["PTS"]  * 1.0) +
    (df["REB"]  * 1.2) +
    (df["AST"]  * 1.5) +
    (df["STL"]  * 2.0) +
    (df["BLK"]  * 1.3) -
    (df["TOV"]  * 2.0)
) * df["FG_PCT"]

# Longevity bonus — your idea, mathematically implemented
# Multiply by log of games played so more games = more value
# but with diminishing returns (82 games isn't 2x better than 41)
import numpy as np
df["Value_Score"] = (df["Quality_Score"] * np.log(df["GP"])).round(2)
df["Quality_Score"] = df["Quality_Score"].round(4)

# Sort by Quality first
df_quality = df.sort_values("Quality_Score", ascending=False).head(15)
df_value = df.sort_values("Value_Score", ascending=False).head(15)

# --- SIDE BY SIDE CHART ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Quality chart (per game)
colors1 = ["gold" if i == 0 else "steelblue" for i in range(len(df_quality))]
ax1.barh(df_quality["PLAYER"], df_quality["Quality_Score"], color=colors1)
ax1.invert_yaxis()
ax1.set_title("Quality Score (Per Game)\nHow good is this player?", fontsize=12)
ax1.set_xlabel("Score")

# Value chart (with longevity)
colors2 = ["gold" if i == 0 else "steelblue" for i in range(len(df_value))]
ax2.barh(df_value["PLAYER"], df_value["Value_Score"], color=colors2)
ax2.invert_yaxis()
ax2.set_title("Value Score (Per Game × Longevity)\nHow valuable to their team?", fontsize=12)
ax2.set_xlabel("Score")

plt.suptitle("NBA 2025-26 — Quality vs Value", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("quality_vs_value.png", dpi=150, bbox_inches="tight")
print("Saved as quality_vs_value.png")
plt.show()
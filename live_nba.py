from nba_api.stats.endpoints import leagueleaders
import pandas as pd

print("Fetching live NBA data... please wait")

# Pull current season scoring leaders from the real NBA stats API
leaders = leagueleaders.LeagueLeaders(
    season="2025-26",
    stat_category_abbreviation="PTS"
)

# Convert to a pandas DataFrame
df = leaders.get_data_frames()[0]

# Select only the columns we care about
df = df[["PLAYER", "TEAM", "GP", "PTS", "REB", "AST", "STL", "FG_PCT"]]

# Calculate our efficiency metric from before
df["Efficiency"] = (
    df["PTS"] + df["REB"] + df["AST"] + df["STL"]
) * df["FG_PCT"]

df["Efficiency"] = df["Efficiency"].round(2)

# Sort and show top 15
df = df.sort_values("Efficiency", ascending=False).head(15)

print("\n=== TOP 15 NBA PLAYERS BY EFFICIENCY (2024-25) ===")
print(df.to_string(index=False))
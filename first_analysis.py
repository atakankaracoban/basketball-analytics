import pandas as pd

# We're creating a small dataset of NBA players
# with their basic stats per 36 minutes

players = {
    "Player": ["Nikola Jokic", "Shai Gilgeous-Alexander", "Luka Doncic", "Jayson Tatum", "Anthony Davis"],
    "Points": [26.4, 31.4, 28.7, 24.9, 24.1],
    "Rebounds": [12.4, 5.5, 8.5, 8.1, 12.6],
    "Assists": [9.0, 6.1, 8.7, 4.9, 3.5],
    "Steals": [1.4, 2.0, 1.2, 1.1, 1.3],
    "Turnovers": [3.6, 2.8, 3.5, 2.4, 2.1],
    "FG_Percent": [0.583, 0.535, 0.496, 0.457, 0.556]
}

# Turn it into a DataFrame — think of this as a smart spreadsheet
df = pd.DataFrame(players)

# Now let's calculate a simplified efficiency score
# This is YOUR first custom basketball metric
df["Efficiency"] = (
    df["Points"] +
    df["Rebounds"] +
    df["Assists"] +
    df["Steals"] -
    df["Turnovers"]
) * df["FG_Percent"]

# Round to 2 decimal places
df["Efficiency"] = df["Efficiency"].round(2)

# Sort by efficiency, best first
df = df.sort_values("Efficiency", ascending=False)

print("=== YOUR FIRST BASKETBALL ANALYSIS ===")
print(df[["Player", "Efficiency"]].to_string(index=False))
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd
import matplotlib.pyplot as plt

# --- GAME SCORE FUNCTION ---
# Now we have our own implementation of the correct formula
def calculate_game_score(row):
    return (
        row["PTS"]
        + (0.4 * row["FGM"])
        - (0.7 * row["FGA"])
        - (0.4 * (row["FTA"] - row["FTM"]))
        + (0.7 * row["OREB"])
        + (0.3 * row["DREB"])
        + row["STL"]
        + (0.7 * row["AST"])
        + (0.7 * row["BLK"])
        - (0.4 * row["PF"])
        - row["TOV"]
    )

# --- FIND A PLAYER BY NAME ---
def get_player_id(name):
    all_players = players.get_players()
    for p in all_players:
        if p["full_name"].lower() == name.lower():
            return p["id"]
    return None

# --- CHANGE THIS NAME TO ANY PLAYER YOU WANT ---
player_name = "Bam Adebayo"

print(f"Fetching game log for {player_name}...")
player_id = get_player_id(player_name)

if not player_id:
    print("Player not found. Check the spelling.")
else:
    gamelog = playergamelog.PlayerGameLog(
        player_id=player_id,
        season="2025-26"
    )

    df = gamelog.get_data_frames()[0]

    # Calculate Game Score for every game this season
    df["GAME_SCORE"] = df.apply(calculate_game_score, axis=1).round(2)

    # Sort by date (most recent first)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE")

    # --- CHART ---
    plt.figure(figsize=(14, 6))

    # Color bars by performance level
    colors = []
    for gs in df["GAME_SCORE"]:
        if gs >= 30:
            colors.append("gold")       # Historic
        elif gs >= 20:
            colors.append("steelblue")  # Great
        elif gs >= 10:
            colors.append("cadetblue")  # Solid
        else:
            colors.append("salmon")     # Poor

    plt.bar(range(len(df)), df["GAME_SCORE"], color=colors)

    # Add average line
    avg = df["GAME_SCORE"].mean()
    plt.axhline(y=avg, color="white", linestyle="--", linewidth=1.5,
                label=f"Season Average: {avg:.1f}")

    plt.xlabel("Game Number", fontsize=11)
    plt.ylabel("Game Score", fontsize=11)
    plt.title(f"{player_name} — Game Score Every Game (2025-26)", fontsize=13)
    plt.legend()

    # Add color legend as text
    plt.text(0.01, 0.95, "🟡 ≥30 Historic  🔵 ≥20 Great  🩵 ≥10 Solid  🔴 <10 Poor",
             transform=plt.gca().transAxes, fontsize=8, verticalalignment="top")

    plt.tight_layout()
    plt.savefig("game_score.png", dpi=150, bbox_inches="tight")
    print(f"Season average Game Score: {avg:.2f}")
    print(f"Best game: {df['GAME_SCORE'].max()}")
    print(f"Games with GS ≥ 20: {len(df[df['GAME_SCORE'] >= 20])}")
    plt.show()
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd
import matplotlib.pyplot as plt

# --- GET PLAYER DATA ---
def get_player_id(name):
    all_players = players.get_players()
    for p in all_players:
        if p["full_name"].lower() == name.lower():
            return p["id"]
    return None

player_name = "Kon Knueppel"

print(f"Fetching data for {player_name}...")
player_id = get_player_id(player_name)

gamelog = playergamelog.PlayerGameLog(
    player_id=player_id,
    season="2025-26"
)

df = gamelog.get_data_frames()[0]
df = df.sort_values("GAME_DATE").reset_index(drop=True)

# --- THIS IS THE CORE LESSON ---
# cumsum() means "running total" — add each game to everything before it
# This is a LOOP happening inside pandas automatically
df["cumulative_fga"] = df["FGA"].cumsum()      # Total attempts so far
df["cumulative_fgm"] = df["FGM"].cumsum()      # Total makes so far
df["cumulative_fg_pct"] = (
    df["cumulative_fgm"] / df["cumulative_fga"]
).round(4)

# Same for three pointers
df["cumulative_fg3a"] = df["FG3A"].cumsum()
df["cumulative_fg3m"] = df["FG3M"].cumsum()
df["cumulative_fg3_pct"] = (
    df["cumulative_fg3m"] / df["cumulative_fg3a"]
).round(4)

# --- NOW A REAL LOOP WITH CONDITIONALS ---
# For each game, we check if we've hit the reliability threshold
# This is you learning loops and if/else in a meaningful context

fg_reliable_game = None      # Which game did FG% become trustworthy?
fg3_reliable_game = None     # Which game did 3P% become trustworthy?

for i in range(len(df)):                        # Loop through every game
    attempts_so_far = df["cumulative_fga"][i]   # How many FG attempts total?
    threes_so_far = df["cumulative_fg3a"][i]    # How many 3PT attempts total?

    if attempts_so_far >= 150:                  # Conditional — hit threshold?
        if fg_reliable_game is None:            # Only record the FIRST time
            fg_reliable_game = i

    if threes_so_far >= 300:                    # 3PT needs more shots
        if fg3_reliable_game is None:
            fg3_reliable_game = i

# --- CHART ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# --- TOP CHART: FG% stabilization ---
colors_fg = []
for i in range(len(df)):
    if fg_reliable_game and i >= fg_reliable_game:
        colors_fg.append("steelblue")    # Trustworthy zone
    else:
        colors_fg.append("salmon")       # Too early to trust

ax1.bar(range(len(df)), df["cumulative_fg_pct"],
        color=colors_fg, alpha=0.7)
ax1.axhline(y=df["cumulative_fg_pct"].iloc[-1],
            color="white", linestyle="--", linewidth=1.5,
            label=f"Final FG%: {df['cumulative_fg_pct'].iloc[-1]:.3f}")

if fg_reliable_game:
    ax1.axvline(x=fg_reliable_game, color="yellow",
                linestyle="--", linewidth=2,
                label=f"Reliable after game {fg_reliable_game}")

ax1.set_title(f"{player_name} — Field Goal % Stabilization", fontsize=12)
ax1.set_ylabel("Cumulative FG%")
ax1.legend()
ax1.text(0.01, 0.05, "🔴 Too few attempts — unreliable   🔵 150+ attempts — trustworthy",
         transform=ax1.transAxes, fontsize=9)

# --- BOTTOM CHART: 3PT% stabilization ---
colors_3pt = []
for i in range(len(df)):
    if fg3_reliable_game and i >= fg3_reliable_game:
        colors_3pt.append("steelblue")
    else:
        colors_3pt.append("salmon")

ax2.bar(range(len(df)), df["cumulative_fg3_pct"],
        color=colors_3pt, alpha=0.7)
ax2.axhline(y=df["cumulative_fg3_pct"].iloc[-1],
            color="white", linestyle="--", linewidth=1.5,
            label=f"Final 3PT%: {df['cumulative_fg3_pct'].iloc[-1]:.3f}")

if fg3_reliable_game:
    ax2.axvline(x=fg3_reliable_game, color="yellow",
                linestyle="--", linewidth=2,
                label=f"Reliable after game {fg3_reliable_game}")
else:
    ax2.text(0.3, 0.5, "NOT YET RELIABLE THIS SEASON",
             transform=ax2.transAxes, fontsize=14,
             color="salmon", fontweight="bold")

ax2.set_title(f"{player_name} — Three Point % Stabilization", fontsize=12)
ax2.set_ylabel("Cumulative 3PT%")
ax2.set_xlabel("Game Number")
ax2.legend()
ax2.text(0.01, 0.05, "🔴 Too few attempts — unreliable   🔵 300+ attempts — trustworthy",
         transform=ax2.transAxes, fontsize=9)

plt.suptitle("Sample Size Reliability — When Can We Trust The Numbers?",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("sample_size.png", dpi=150, bbox_inches="tight")
print("Saved as sample_size.png")

# Print key findings
print(f"\n--- {player_name} Sample Size Report ---")
print(f"Total FG attempts: {df['cumulative_fga'].iloc[-1]}")
print(f"Total 3PT attempts: {df['cumulative_fg3a'].iloc[-1]}")
print(f"FG% reliable after game: {fg_reliable_game if fg_reliable_game else 'Not yet'}")
print(f"3PT% reliable after game: {fg3_reliable_game if fg3_reliable_game else 'Not yet'}")
print(f"Final FG%: {df['cumulative_fg_pct'].iloc[-1]:.3f}")
print(f"Final 3PT%: {df['cumulative_fg3_pct'].iloc[-1]:.3f}")

plt.show()
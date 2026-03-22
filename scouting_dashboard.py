from nba_api.stats.endpoints import playergamelog, playercareerstats, leaguedashplayerstats
from nba_api.stats.static import players
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats
import time

# ============================================================
# CHANGE THIS TO ANY PLAYER YOU WANT
# ============================================================
PLAYER_NAME = "Nikola Jokić"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_player_id(name):
    all_players = players.get_players()
    for p in all_players:
        if p["full_name"].lower() == name.lower():
            return p["id"]
    return None

def calculate_game_score(row):
    return round(
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
        - row["TOV"], 2
    )

def get_percentile(value, series):
    """How does this value rank among all players? Returns 0-100."""
    return round(stats.percentileofscore(series.dropna(), value), 1)

# ============================================================
# FETCH ALL DATA
# ============================================================
print(f"Building scouting dashboard for {PLAYER_NAME}...")
print("Fetching data — this may take 20-30 seconds...\n")

player_id = get_player_id(PLAYER_NAME)
if not player_id:
    print(f"Player '{PLAYER_NAME}' not found. Check spelling.")
    exit()

# 1. Current season game log
time.sleep(0.5)
gamelog = playergamelog.PlayerGameLog(
    player_id=player_id,
    season="2025-26"
)
df_games = gamelog.get_data_frames()[0]
df_games = df_games.sort_values("GAME_DATE").reset_index(drop=True)

# 2. Career stats
time.sleep(0.5)
career = playercareerstats.PlayerCareerStats(player_id=player_id)
df_career = career.get_data_frames()[0]
df_career = df_career[df_career["GP"] >= 20].copy()

# 3. League-wide stats for percentile comparison
time.sleep(0.5)
league = leaguedashplayerstats.LeagueDashPlayerStats(
    season="2025-26",
    per_mode_detailed="PerGame"
)
df_league = league.get_data_frames()[0]
df_league = df_league[df_league["GP"] >= 20]

# Find this player in league stats
player_row = df_league[df_league["PLAYER_ID"] == player_id]
if player_row.empty:
    print("Player not found in league stats.")
    exit()
player_stats = player_row.iloc[0]

# ============================================================
# CALCULATE GAME SCORES
# ============================================================
df_games["GAME_SCORE"] = df_games.apply(calculate_game_score, axis=1)

# ============================================================
# CAREER TRAJECTORY — per game stats by season
# ============================================================
df_career["PPG"] = (df_career["PTS"] / df_career["GP"]).round(1)
df_career["RPG"] = (df_career["REB"] / df_career["GP"]).round(1)
df_career["APG"] = (df_career["AST"] / df_career["GP"]).round(1)
df_career["SEASON"] = df_career["SEASON_ID"].str[:4].astype(int)

# ============================================================
# SHOT RELIABILITY — cumulative FG%
# ============================================================
df_games["CUM_FGA"] = df_games["FGA"].cumsum()
df_games["CUM_FGM"] = df_games["FGM"].cumsum()
df_games["CUM_FG_PCT"] = (df_games["CUM_FGM"] / df_games["CUM_FGA"]).round(4)

reliable_game = None
for i in range(len(df_games)):
    if df_games["CUM_FGA"][i] >= 150 and reliable_game is None:
        reliable_game = i

# ============================================================
# SCOUTING SUMMARY — auto-generated text
# ============================================================
ppg = player_stats["PTS"]
rpg = player_stats["REB"]
apg = player_stats["AST"]
spg = player_stats["STL"]
bpg = player_stats["BLK"]
fg_pct = player_stats["FG_PCT"]
tov = player_stats["TOV"]
gp = player_stats["GP"]

ppg_pct = get_percentile(ppg, df_league["PTS"])
rpg_pct = get_percentile(rpg, df_league["REB"])
apg_pct = get_percentile(apg, df_league["AST"])
gs_avg = df_games["GAME_SCORE"].mean()
gs_max = df_games["GAME_SCORE"].max()

# Determine player type automatically
strengths = []
weaknesses = []

if ppg_pct >= 75: strengths.append("elite scorer")
elif ppg_pct >= 50: strengths.append("above-average scorer")
else: weaknesses.append("limited scoring")

if rpg_pct >= 75: strengths.append("dominant rebounder")
elif rpg_pct >= 50: strengths.append("solid rebounder")

if apg_pct >= 75: strengths.append("elite playmaker")
elif apg_pct >= 50: strengths.append("good passer")

if fg_pct >= 0.52: strengths.append("highly efficient shooter")
elif fg_pct <= 0.44: weaknesses.append("inefficient shooting")

if tov >= 3.5: weaknesses.append("turnover-prone")
elif tov <= 2.0: strengths.append("excellent ball security")

summary_lines = [
    f"SCOUTING REPORT: {PLAYER_NAME}",
    f"{'─' * 35}",
    f"Games Played:  {int(gp)}",
    f"Avg Game Score: {gs_avg:.1f}  |  Peak: {gs_max:.1f}",
    f"",
    f"PERCENTILE RANKINGS:",
    f"  Scoring:    {ppg_pct:.0f}th percentile",
    f"  Rebounding: {rpg_pct:.0f}th percentile",
    f"  Playmaking: {apg_pct:.0f}th percentile",
    f"",
    f"STRENGTHS:",
]
for s in strengths:
    summary_lines.append(f"  + {s.title()}")

if weaknesses:
    summary_lines.append(f"")
    summary_lines.append(f"CONCERNS:")
    for w in weaknesses:
        summary_lines.append(f"  - {w.title()}")

summary_lines.append(f"")
summary_lines.append(f"VERDICT:")
if gs_avg >= 25:
    summary_lines.append(f"  MVP-caliber season.")
elif gs_avg >= 18:
    summary_lines.append(f"  All-Star caliber season.")
elif gs_avg >= 12:
    summary_lines.append(f"  Solid starter.")
else:
    summary_lines.append(f"  Rotation player.")

# ============================================================
# BUILD THE DASHBOARD
# ============================================================
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("#0d1117")

gs_layout = gridspec.GridSpec(3, 3, figure=fig,
                               hspace=0.45, wspace=0.35)

fig.suptitle(f"NBA SCOUTING DASHBOARD — {PLAYER_NAME.upper()}  |  2025-26 Season",
             fontsize=16, fontweight="bold", color="white", y=0.98)

# ── Panel 1: Season Stats Box ──────────────────────────────
ax1 = fig.add_subplot(gs_layout[0, 0])
ax1.set_facecolor("#161b22")
ax1.axis("off")

stat_text = (
    f"  {'SEASON AVERAGES':^28}\n"
    f"  {'─' * 28}\n\n"
    f"  Points:      {ppg:.1f} ppg\n"
    f"  Rebounds:    {rpg:.1f} rpg\n"
    f"  Assists:     {apg:.1f} apg\n"
    f"  Steals:      {spg:.1f} spg\n"
    f"  Blocks:      {bpg:.1f} bpg\n"
    f"  Turnovers:   {tov:.1f} tpg\n\n"
    f"  FG%:         {fg_pct:.3f}\n"
    f"  Games:       {int(gp)}\n"
)
ax1.text(0.05, 0.95, stat_text, transform=ax1.transAxes,
         fontsize=11, verticalalignment="top", color="white",
         fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="#1f2937", alpha=0.8))
ax1.set_title("Season Stats", color="white", fontsize=11, pad=8)

# ── Panel 2: Percentile Bars ───────────────────────────────
ax2 = fig.add_subplot(gs_layout[0, 1])
ax2.set_facecolor("#161b22")

pct_stats = ["PTS", "REB", "AST", "STL", "BLK"]
pct_labels = ["Points", "Rebounds", "Assists", "Steals", "Blocks"]
pct_values = [get_percentile(player_stats[s], df_league[s])
              for s in pct_stats]

colors_pct = ["gold" if v >= 75 else "steelblue" if v >= 50
              else "salmon" for v in pct_values]
bars = ax2.barh(pct_labels, pct_values, color=colors_pct, alpha=0.85)

for bar, val in zip(bars, pct_values):
    ax2.text(min(val + 1, 95), bar.get_y() + bar.get_height()/2,
             f"{val:.0f}th", va="center", color="white", fontsize=9)

ax2.axvline(x=50, color="gray", linestyle="--", alpha=0.5)
ax2.set_xlim(0, 100)
ax2.set_xlabel("Percentile", color="white", fontsize=9)
ax2.set_title("League Percentile Rankings\n(Gold = Top 25%)",
              color="white", fontsize=11)
ax2.tick_params(colors="white")
for spine in ax2.spines.values():
    spine.set_edgecolor("#333366")

# ── Panel 3: Game Score Timeline ──────────────────────────
ax3 = fig.add_subplot(gs_layout[1, :2])
ax3.set_facecolor("#161b22")

gs_colors = []
for gs in df_games["GAME_SCORE"]:
    if gs >= 30: gs_colors.append("gold")
    elif gs >= 20: gs_colors.append("steelblue")
    elif gs >= 10: gs_colors.append("cadetblue")
    else: gs_colors.append("salmon")

ax3.bar(range(len(df_games)), df_games["GAME_SCORE"],
        color=gs_colors, alpha=0.85)
ax3.axhline(y=gs_avg, color="white", linestyle="--",
            linewidth=1.5, label=f"Avg: {gs_avg:.1f}")
ax3.set_title("Game Score — Every Game This Season",
              color="white", fontsize=11)
ax3.set_xlabel("Game Number", color="white", fontsize=9)
ax3.set_ylabel("Game Score", color="white", fontsize=9)
ax3.legend(facecolor="#161b22", labelcolor="white", fontsize=9)
ax3.tick_params(colors="white")
ax3.text(0.01, 0.95,
         "🟡 ≥30 Historic  🔵 ≥20 Great  🩵 ≥10 Solid  🔴 <10 Poor",
         transform=ax3.transAxes, fontsize=8,
         verticalalignment="top", color="white")
for spine in ax3.spines.values():
    spine.set_edgecolor("#333366")

# ── Panel 4: Career Trajectory ────────────────────────────
ax4 = fig.add_subplot(gs_layout[2, :2])
ax4.set_facecolor("#161b22")

ax4.plot(df_career["SEASON"], df_career["PPG"],
         "o-", color="gold", linewidth=2, label="Points", markersize=5)
ax4.plot(df_career["SEASON"], df_career["RPG"],
         "s-", color="steelblue", linewidth=2,
         label="Rebounds", markersize=5)
ax4.plot(df_career["SEASON"], df_career["APG"],
         "^-", color="lightgreen", linewidth=2,
         label="Assists", markersize=5)

ax4.set_title("Career Trajectory — Key Stats by Season",
              color="white", fontsize=11)
ax4.set_xlabel("Season Start Year", color="white", fontsize=9)
ax4.set_ylabel("Per Game", color="white", fontsize=9)
ax4.legend(facecolor="#161b22", labelcolor="white", fontsize=9)
ax4.tick_params(colors="white")
ax4.grid(color="gray", alpha=0.2)
for spine in ax4.spines.values():
    spine.set_edgecolor("#333366")

# ── Panel 5: Shot Reliability ─────────────────────────────
ax5 = fig.add_subplot(gs_layout[0, 2])
ax5.set_facecolor("#161b22")

rel_colors = []
for i in range(len(df_games)):
    if reliable_game and i >= reliable_game:
        rel_colors.append("steelblue")
    else:
        rel_colors.append("salmon")

ax5.bar(range(len(df_games)), df_games["CUM_FG_PCT"],
        color=rel_colors, alpha=0.8)

if reliable_game:
    ax5.axvline(x=reliable_game, color="yellow",
                linestyle="--", linewidth=1.5,
                label=f"Reliable: game {reliable_game}")
    ax5.legend(facecolor="#161b22", labelcolor="white", fontsize=8)

ax5.set_title("FG% Reliability\n(Red=Noise, Blue=Trustworthy)",
              color="white", fontsize=10)
ax5.set_xlabel("Game", color="white", fontsize=8)
ax5.set_ylabel("Cumulative FG%", color="white", fontsize=8)
ax5.tick_params(colors="white", labelsize=7)
for spine in ax5.spines.values():
    spine.set_edgecolor("#333366")

# ── Panel 6: Scouting Summary ─────────────────────────────
ax6 = fig.add_subplot(gs_layout[1:, 2])
ax6.set_facecolor("#161b22")
ax6.axis("off")

summary_text = "\n".join(summary_lines)
ax6.text(0.05, 0.97, summary_text, transform=ax6.transAxes,
         fontsize=9.5, verticalalignment="top", color="white",
         fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="#1f2937", alpha=0.8))

# ── Save ──────────────────────────────────────────────────
filename = f"dashboard_{PLAYER_NAME.replace(' ', '_')}.png"
plt.savefig(filename, dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
print(f"Dashboard saved as {filename}")
plt.show()
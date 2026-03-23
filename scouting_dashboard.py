from nba_api.stats.endpoints import playergamelog, playercareerstats, leaguedashplayerstats
from nba_api.stats.static import players
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats
import time

PLAYER_NAME = "Nikola Jokić"

def get_player_id(name):
    all_players = players.get_players()
    for p in all_players:
        if p["full_name"].lower() == name.lower():
            return p["id"]
    return None

def calculate_game_score(row):
    return round(
        row["PTS"] + (0.4 * row["FGM"]) - (0.7 * row["FGA"])
        - (0.4 * (row["FTA"] - row["FTM"]))
        + (0.7 * row["OREB"]) + (0.3 * row["DREB"])
        + row["STL"] + (0.7 * row["AST"])
        + (0.7 * row["BLK"]) - (0.4 * row["PF"]) - row["TOV"], 2
    )

def get_percentile(value, series):
    return round(stats.percentileofscore(series.dropna(), value), 1)

def get_league_stats(season_str):
    try:
        league = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season_str,
            per_mode_detailed="PerGame"
        )
        df = league.get_data_frames()[0]
        return df[df["GP"] >= 20]
    except:
        return pd.DataFrame()

def find_player_season(player_id, peak_season_str):
    """Try peak season and adjacent seasons until we find the player."""
    seasons_to_try = [peak_season_str]
    year = int(peak_season_str[:4])
    for adj in range(1, 15):
        for direction in [-1, 1]:
            adj_year = year + (adj * direction)
            adj_season = f"{adj_year}-{str(adj_year + 1)[-2:]}"
            seasons_to_try.append(adj_season)

    for season in seasons_to_try:
        print(f"Trying {season}...")
        time.sleep(0.3)
        df_league = get_league_stats(season)
        if df_league.empty:
            continue
        player_row = df_league[df_league["PLAYER_ID"] == player_id]
        if not player_row.empty:
            print(f"Found in {season}")
            return df_league, player_row, season
    return None, None, None

# ── Fetch Data ─────────────────────────────────────────────
print(f"Building scouting dashboard for {PLAYER_NAME}...")

player_id = get_player_id(PLAYER_NAME)
if not player_id:
    print(f"Player '{PLAYER_NAME}' not found.")
    exit()

time.sleep(0.5)
career = playercareerstats.PlayerCareerStats(player_id=player_id)
df_career = career.get_data_frames()[0]
df_career = df_career[df_career["GP"] >= 20].copy()
df_career["FG_PCT"] = df_career["FGM"] / df_career["FGA"]

df_career["PEAK_SCORE"] = (
    (df_career["PTS"] / df_career["GP"] * 1.0) +
    (df_career["REB"] / df_career["GP"] * 1.2) +
    (df_career["AST"] / df_career["GP"] * 1.5) +
    (df_career["STL"] / df_career["GP"] * 2.0) +
    (df_career["BLK"] / df_career["GP"] * 1.3)
) * df_career["FG_PCT"]

best_season_row = df_career.loc[df_career["PEAK_SCORE"].idxmax()]
peak_season_str = best_season_row["SEASON_ID"]
print(f"Peak season identified: {peak_season_str}")

# Try current season first
time.sleep(0.5)
print("Checking current season...")
df_league = get_league_stats("2025-26")
player_row = df_league[df_league["PLAYER_ID"] == player_id]
era_note = "(vs 2025-26 season)"
game_season = "2025-26"

if player_row.empty:
    print(f"Not in current season. Searching historical data...")
    df_league, player_row, found_season = find_player_season(
        player_id, peak_season_str
    )
    if player_row is None:
        print("Could not find player in any season.")
        exit()
    era_note = f"(vs {found_season} era)"
    game_season = found_season

player_stats = player_row.iloc[0]
print(f"Comparing {PLAYER_NAME} {era_note}")

# Game log
time.sleep(0.5)
gamelog = playergamelog.PlayerGameLog(
    player_id=player_id,
    season=game_season
)
df_games = gamelog.get_data_frames()[0]
df_games = df_games.sort_values("GAME_DATE").reset_index(drop=True)

# ── Calculations ───────────────────────────────────────────
df_games["GAME_SCORE"] = df_games.apply(calculate_game_score, axis=1)

df_career["PPG"] = (df_career["PTS"] / df_career["GP"]).round(1)
df_career["RPG"] = (df_career["REB"] / df_career["GP"]).round(1)
df_career["APG"] = (df_career["AST"] / df_career["GP"]).round(1)
df_career["SEASON"] = df_career["SEASON_ID"].str[:4].astype(int)

df_games["CUM_FGA"] = df_games["FGA"].cumsum()
df_games["CUM_FGM"] = df_games["FGM"].cumsum()
df_games["CUM_FG_PCT"] = (df_games["CUM_FGM"] / df_games["CUM_FGA"]).round(4)

reliable_game = None
for i in range(len(df_games)):
    if df_games["CUM_FGA"][i] >= 150 and reliable_game is None:
        reliable_game = i

# ── Scouting Summary ───────────────────────────────────────
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
    f"Season: {game_season}  |  Games: {int(gp)}",
    f"Avg Game Score: {gs_avg:.1f}  |  Peak: {gs_max:.1f}",
    f"",
    f"PERCENTILE RANKINGS {era_note}:",
    f"  Scoring:    {ppg_pct:.0f}th percentile",
    f"  Rebounding: {rpg_pct:.0f}th percentile",
    f"  Playmaking: {apg_pct:.0f}th percentile",
    f"",
    f"STRENGTHS:",
]
for s in strengths:
    summary_lines.append(f"  + {s.title()}")
if weaknesses:
    summary_lines.append("")
    summary_lines.append("CONCERNS:")
    for w in weaknesses:
        summary_lines.append(f"  - {w.title()}")
# Era-adjusted verdict
# Calculate league average game score for context
league_gs_avg = (
    (df_league["PTS"] * 1.0) +
    (df_league["REB"] * 1.2) +
    (df_league["AST"] * 1.5) +
    (df_league["STL"] * 2.0) +
    (df_league["BLK"] * 1.3)
) * df_league["FG_PCT"]
league_gs_mean = league_gs_avg.mean()
gs_above_avg = gs_avg - league_gs_mean

summary_lines.append("")
summary_lines.append(f"League avg score: {league_gs_mean:.1f}")
summary_lines.append(f"Player above avg: +{gs_above_avg:.1f}")
summary_lines.append("")
summary_lines.append("VERDICT:")
if gs_above_avg >= 8: summary_lines.append("  MVP-caliber season.")
elif gs_above_avg >= 5: summary_lines.append("  All-Star caliber season.")
elif gs_above_avg >= 2: summary_lines.append("  Solid starter.")
else: summary_lines.append("  Rotation player.")

# ── Build Dashboard ────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("#0d1117")
gs_layout = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle(
    f"NBA SCOUTING DASHBOARD — {PLAYER_NAME.upper()}  |  {game_season}",
    fontsize=16, fontweight="bold", color="white", y=0.98
)

# Panel 1 — Season Stats
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

# Panel 2 — Percentile Bars
ax2 = fig.add_subplot(gs_layout[0, 1])
ax2.set_facecolor("#161b22")
pct_stats = ["PTS", "REB", "AST", "STL", "BLK"]
pct_labels = ["Points", "Rebounds", "Assists", "Steals", "Blocks"]
pct_values = [get_percentile(player_stats[s], df_league[s]) for s in pct_stats]
colors_pct = ["gold" if v >= 75 else "steelblue" if v >= 50 else "salmon"
              for v in pct_values]
bars = ax2.barh(pct_labels, pct_values, color=colors_pct, alpha=0.85)
for bar, val in zip(bars, pct_values):
    ax2.text(min(val + 1, 95), bar.get_y() + bar.get_height()/2,
             f"{val:.0f}th", va="center", color="white", fontsize=9)
ax2.axvline(x=50, color="gray", linestyle="--", alpha=0.5)
ax2.set_xlim(0, 100)
ax2.set_xlabel("Percentile", color="white", fontsize=9)
ax2.set_title(f"League Percentile Rankings\n{era_note} — Gold = Top 25%",
              color="white", fontsize=11)
ax2.tick_params(colors="white")
for spine in ax2.spines.values():
    spine.set_edgecolor("#333366")

# Panel 3 — Game Score Timeline
ax3 = fig.add_subplot(gs_layout[1, :2])
ax3.set_facecolor("#161b22")
gs_colors = ["gold" if g >= 30 else "steelblue" if g >= 20
             else "cadetblue" if g >= 10 else "salmon"
             for g in df_games["GAME_SCORE"]]
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

# Panel 4 — Career Trajectory
ax4 = fig.add_subplot(gs_layout[2, :2])
ax4.set_facecolor("#161b22")
ax4.plot(df_career["SEASON"], df_career["PPG"],
         "o-", color="gold", linewidth=2, label="Points", markersize=5)
ax4.plot(df_career["SEASON"], df_career["RPG"],
         "s-", color="steelblue", linewidth=2, label="Rebounds", markersize=5)
ax4.plot(df_career["SEASON"], df_career["APG"],
         "^-", color="lightgreen", linewidth=2, label="Assists", markersize=5)
ax4.set_title("Career Trajectory — Key Stats by Season",
              color="white", fontsize=11)
ax4.set_xlabel("Season Start Year", color="white", fontsize=9)
ax4.set_ylabel("Per Game", color="white", fontsize=9)
ax4.legend(facecolor="#161b22", labelcolor="white", fontsize=9)
ax4.tick_params(colors="white")
ax4.grid(color="gray", alpha=0.2)
for spine in ax4.spines.values():
    spine.set_edgecolor("#333366")

# Panel 5 — Shot Reliability
ax5 = fig.add_subplot(gs_layout[0, 2])
ax5.set_facecolor("#161b22")
rel_colors = ["steelblue" if (reliable_game and i >= reliable_game)
              else "salmon" for i in range(len(df_games))]
ax5.bar(range(len(df_games)), df_games["CUM_FG_PCT"],
        color=rel_colors, alpha=0.8)
if reliable_game:
    ax5.axvline(x=reliable_game, color="yellow", linestyle="--",
                linewidth=1.5, label=f"Reliable: game {reliable_game}")
    ax5.legend(facecolor="#161b22", labelcolor="white", fontsize=8)
ax5.set_title("FG% Reliability\n(Red=Noise, Blue=Trustworthy)",
              color="white", fontsize=10)
ax5.set_xlabel("Game", color="white", fontsize=8)
ax5.set_ylabel("Cumulative FG%", color="white", fontsize=8)
ax5.tick_params(colors="white", labelsize=7)
for spine in ax5.spines.values():
    spine.set_edgecolor("#333366")

# Panel 6 — Scouting Summary
ax6 = fig.add_subplot(gs_layout[1:, 2])
ax6.set_facecolor("#161b22")
ax6.axis("off")
ax6.text(0.05, 0.97, "\n".join(summary_lines),
         transform=ax6.transAxes, fontsize=9.5,
         verticalalignment="top", color="white",
         fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="#1f2937", alpha=0.8))

# Save
filename = f"dashboard_{PLAYER_NAME.replace(' ', '_')}.png"
plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="#0d1117")
print(f"Dashboard saved as {filename}")
plt.show()
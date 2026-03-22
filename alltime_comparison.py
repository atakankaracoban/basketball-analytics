from nba_api.stats.endpoints import playercareerstats, leaguedashplayerstats
from nba_api.stats.static import players
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

# ============================================================
# HELPER FUNCTIONS — this is you learning functions deeply
# ============================================================

def get_player_id(name):
    """Find a player's NBA ID by their full name."""
    all_players = players.get_players()
    for p in all_players:
        if p["full_name"].lower() == name.lower():
            return p["id"], p["is_active"]
    return None, None

def get_career_per36(player_id):
    """
    Get career stats normalized to per-36-minutes.
    Per-36 eliminates the advantage of playing more minutes.
    A player who scores 18 pts in 36 min is equal to one
    who scores 18 pts in 36 min — regardless of era.
    """
    career = playercareerstats.PlayerCareerStats(player_id=player_id)
    df = career.get_data_frames()[0]  # Regular season totals

    # Filter out seasons with too few games (sample size!)
    df = df[df["GP"] >= 20]

    if df.empty:
        return None

    # Sum across all seasons for career totals
    totals = df[["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FGA", "FGM", "FG3A", "FG3M", "FTA", "FTM"]].sum()

    # Convert to per-36-minutes
    # This is the key normalization — everything scaled to 36 minutes
    per36 = {}
    if totals["MIN"] > 0:
        factor = 36 / totals["MIN"] * len(df[df["GP"] >= 20])
        # Actually calculate properly game by game
        # Recalculate: per36 = (stat / total_minutes) * 36 * games
        mins_per_game = totals["MIN"] / df["GP"].sum()
        scale = 36 / mins_per_game if mins_per_game > 0 else 1

        per36["PTS"]  = round((totals["PTS"]  / df["GP"].sum()) * (36 / mins_per_game), 1)
        per36["REB"]  = round((totals["REB"]  / df["GP"].sum()) * (36 / mins_per_game), 1)
        per36["AST"]  = round((totals["AST"]  / df["GP"].sum()) * (36 / mins_per_game), 1)
        per36["STL"]  = round((totals["STL"]  / df["GP"].sum()) * (36 / mins_per_game), 1)
        per36["BLK"]  = round((totals["BLK"]  / df["GP"].sum()) * (36 / mins_per_game), 1)
        per36["TOV"]  = round((totals["TOV"]  / df["GP"].sum()) * (36 / mins_per_game), 1)
        per36["FG_PCT"] = round(totals["FGM"] / totals["FGA"], 3) if totals["FGA"] > 0 else 0
        per36["FG3_PCT"] = round(totals["FG3M"] / totals["FG3A"], 3) if totals["FG3A"] > 0 else 0
        per36["FT_PCT"] = round(totals["FTM"] / totals["FTA"], 3) if totals["FTA"] > 0 else 0
        per36["GP"] = df["GP"].sum()
        per36["SEASONS"] = len(df)

    return per36

def get_peak_season(player_id):
    """
    Find a player's single best season by Game Score average.
    Peak matters — some players had transcendent single seasons.
    """
    career = playercareerstats.PlayerCareerStats(player_id=player_id)
    df = career.get_data_frames()[0]
    df = df[df["GP"] >= 30]  # Minimum sample size for a season

    if df.empty:
        return None

    # Calculate simple efficiency per game for each season
    df = df.copy()
    df["MPG"] = df["MIN"] / df["GP"]
    df["PPG"] = df["PTS"] / df["GP"]
    df["RPG"] = df["REB"] / df["GP"]
    df["APG"] = df["AST"] / df["GP"]
    df["SPG"] = df["STL"] / df["GP"]
    df["BPG"] = df["BLK"] / df["GP"]
    df["TOPG"] = df["TOV"] / df["GP"]
    df["FG_PCT"] = df["FGM"] / df["FGA"]

    # Game Score average as peak metric
    df["PEAK_SCORE"] = (
        df["PPG"] * 1.0 +
        df["RPG"] * 1.2 +
        df["APG"] * 1.5 +
        df["SPG"] * 2.0 +
        df["BPG"] * 1.3 -
        df["TOPG"] * 2.0
    ) * df["FG_PCT"]
    # Steals weighted at 2.0 — justified by full possession swing value
# (prevents opponent score + gains possession = ~2 point swing)
# Alternative: 1.5 if concerned about defensive gambling bias

    best = df.loc[df["PEAK_SCORE"].idxmax()]
    return {
        "SEASON": best["SEASON_ID"],
        "PPG": round(best["PPG"], 1),
        "RPG": round(best["RPG"], 1),
        "APG": round(best["APG"], 1),
        "SPG": round(best["SPG"], 1),
        "BPG": round(best["BPG"], 1),
        "FG_PCT": round(best["FG_PCT"], 3),
        "PEAK_SCORE": round(best["PEAK_SCORE"], 2),
        "GP": int(best["GP"])
    }

# ============================================================
# RADAR CHART FUNCTION
# ============================================================

def make_radar_chart(ax, values, labels, color, player_name, max_vals):
    """
    Draw a radar/spider chart for one player.
    Each axis represents one stat, normalized 0-1.
    """
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon

    # Normalize values between 0 and 1 relative to max
    normalized = [min(v / m, 1.0) if m > 0 else 0
                  for v, m in zip(values, max_vals)]
    normalized += normalized[:1]

    ax.plot(angles, normalized, color=color, linewidth=2)
    ax.fill(angles, normalized, color=color, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0, 1)
    ax.set_title(player_name, size=11, color=color, pad=15)
    ax.grid(color="gray", alpha=0.3)

# ============================================================
# MAIN COMPARISON — CHANGE THESE TWO NAMES
# ============================================================

player1_name = "Larry Bird"
player2_name = "Magic Johnson"

print(f"Fetching data for {player1_name} and {player2_name}...")
print("This may take 10-20 seconds — pulling historical data...\n")

p1_id, p1_active = get_player_id(player1_name)
time.sleep(1)  # Respectful pause between API calls
p2_id, p2_active = get_player_id(player2_name)

if not p1_id or not p2_id:
    print("One or both players not found. Check spelling.")
else:
    time.sleep(1)
    p1_career = get_career_per36(p1_id)
    time.sleep(1)
    p2_career = get_career_per36(p2_id)
    time.sleep(1)
    p1_peak = get_peak_season(p1_id)
    time.sleep(1)
    p2_peak = get_peak_season(p2_id)

    # ============================================================
    # BUILD THE VISUALIZATION — 3 panels
    # ============================================================

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#1a1a2e")

    # Title
    fig.suptitle(f"{player1_name}  vs  {player2_name}\nAll-Time Comparison",
                 fontsize=16, fontweight="bold", color="white", y=0.98)

    p1_color = "#4fc3f7"   # Blue for player 1
    p2_color = "#ff8a65"   # Orange for player 2

    # ============================================================
    # PANEL 1 — Career Per-36 Bar Chart (left)
    # ============================================================
    ax1 = fig.add_subplot(2, 3, (1, 2))
    ax1.set_facecolor("#16213e")

    stats_to_show = ["PTS", "REB", "AST", "STL", "BLK"]
    labels_show = ["Points", "Rebounds", "Assists", "Steals", "Blocks"]
    x = np.arange(len(stats_to_show))
    width = 0.35

    p1_vals = [p1_career[s] for s in stats_to_show]
    p2_vals = [p2_career[s] for s in stats_to_show]

    bars1 = ax1.bar(x - width/2, p1_vals, width,
                    label=player1_name, color=p1_color, alpha=0.85)
    bars2 = ax1.bar(x + width/2, p2_vals, width,
                    label=player2_name, color=p2_color, alpha=0.85)

    # Add value labels on bars
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{bar.get_height()}", ha="center", va="bottom",
                 color="white", fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{bar.get_height()}", ha="center", va="bottom",
                 color="white", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_show, color="white")
    ax1.set_ylabel("Per 36 Minutes", color="white")
    ax1.set_title("Career Stats — Per 36 Minutes\n(Eliminates playing time bias)",
                  color="white", fontsize=11)
    ax1.legend(facecolor="#16213e", labelcolor="white")
    ax1.tick_params(colors="white")
    ax1.set_facecolor("#16213e")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333366")

    # ============================================================
    # PANEL 2 — Shooting % Comparison (right of bar chart)
    # ============================================================
    ax2 = fig.add_subplot(2, 3, 3)
    ax2.set_facecolor("#16213e")

    shoot_stats = ["FG_PCT", "FG3_PCT", "FT_PCT"]
    shoot_labels = ["FG%", "3PT%", "FT%"]
    x2 = np.arange(len(shoot_stats))

    p1_shoot = [p1_career[s] for s in shoot_stats]
    p2_shoot = [p2_career[s] for s in shoot_stats]

    ax2.bar(x2 - width/2, p1_shoot, width, color=p1_color, alpha=0.85,
            label=player1_name)
    ax2.bar(x2 + width/2, p2_shoot, width, color=p2_color, alpha=0.85,
            label=player2_name)

    for i, (v1, v2) in enumerate(zip(p1_shoot, p2_shoot)):
        ax2.text(i - width/2, v1 + 0.005, f"{v1:.3f}",
                 ha="center", color="white", fontsize=8)
        ax2.text(i + width/2, v2 + 0.005, f"{v2:.3f}",
                 ha="center", color="white", fontsize=8)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(shoot_labels, color="white")
    ax2.set_title("Career Shooting %", color="white", fontsize=11)
    ax2.set_ylim(0, 0.75)
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#16213e", labelcolor="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333366")

    # ============================================================
    # PANEL 3 — Radar Charts (career)
    # ============================================================
    ax3 = fig.add_subplot(2, 3, 4, polar=True)
    ax4 = fig.add_subplot(2, 3, 5, polar=True)
    ax3.set_facecolor("#16213e")
    ax4.set_facecolor("#16213e")

    radar_stats = ["PTS", "REB", "AST", "STL", "BLK"]
    radar_labels = ["Points", "Rebounds", "Assists", "Steals", "Blocks"]

    p1_radar = [p1_career[s] for s in radar_stats]
    p2_radar = [p2_career[s] for s in radar_stats]

    # Max values for normalization — use the higher of the two players
    max_vals = [max(p1_radar[i], p2_radar[i]) * 1.1
                for i in range(len(radar_stats))]

    make_radar_chart(ax3, p1_radar, radar_labels, p1_color,
                     f"{player1_name}\nCareer Per-36", max_vals)
    make_radar_chart(ax4, p2_radar, radar_labels, p2_color,
                     f"{player2_name}\nCareer Per-36", max_vals)

    # ============================================================
    # PANEL 4 — Peak Season Stats (bottom right)
    # ============================================================
    ax5 = fig.add_subplot(2, 3, 6)
    ax5.set_facecolor("#16213e")
    ax5.axis("off")

    # Build a text summary of peak seasons
    peak_text = (
        f"{'PEAK SEASON COMPARISON':^40}\n"
        f"{'─' * 40}\n\n"
        f"{player1_name} — {p1_peak['SEASON']}\n"
        f"  Points:   {p1_peak['PPG']} ppg\n"
        f"  Rebounds: {p1_peak['RPG']} rpg\n"
        f"  Assists:  {p1_peak['APG']} apg\n"
        f"  Steals:   {p1_peak['SPG']} spg\n"
        f"  Blocks:   {p1_peak['BPG']} bpg\n"
        f"  FG%:      {p1_peak['FG_PCT']}\n"
        f"  Games:    {p1_peak['GP']}\n"
        f"  Peak Score: {p1_peak['PEAK_SCORE']}\n\n"
        f"{player2_name} — {p2_peak['SEASON']}\n"
        f"  Points:   {p2_peak['PPG']} ppg\n"
        f"  Rebounds: {p2_peak['RPG']} rpg\n"
        f"  Assists:  {p2_peak['APG']} apg\n"
        f"  Steals:   {p2_peak['SPG']} spg\n"
        f"  Blocks:   {p2_peak['BPG']} bpg\n"
        f"  FG%:      {p2_peak['FG_PCT']}\n"
        f"  Games:    {p2_peak['GP']}\n"
        f"  Peak Score: {p2_peak['PEAK_SCORE']}\n"
    )

    ax5.text(0.05, 0.95, peak_text, transform=ax5.transAxes,
             fontsize=9, verticalalignment="top", color="white",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#0f3460", alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("alltime_comparison3.png", dpi=150, bbox_inches="tight",
                facecolor="#1a1a2e")
    print("Saved as alltime_comparison3.png")
    plt.show()

    # Print summary
    print(f"\n{'='*50}")
    print(f"CAREER SUMMARY")
    print(f"{'='*50}")
    print(f"{player1_name}: {p1_career['GP']} games, "
          f"{p1_career['SEASONS']} seasons")
    print(f"{player2_name}: {p2_career['GP']} games, "
          f"{p2_career['SEASONS']} seasons")
    print(f"\n{player1_name} Peak: {p1_peak['PEAK_SCORE']} "
          f"(Season {p1_peak['SEASON']})")
    print(f"{player2_name} Peak: {p2_peak['PEAK_SCORE']} "
          f"(Season {p2_peak['SEASON']})")
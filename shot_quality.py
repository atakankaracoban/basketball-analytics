"""
Shot Quality Analyzer
Analyzes shot location data and calculates expected points
Visualizes results as a basketball court heatmap
"""

from nba_api.stats.endpoints import (
    shotchartdetail,
    leaguedashplayerstats
)
from nba_api.stats.static import players, teams
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, Arc
import numpy as np
import time

# ============================================================
# DRAW BASKETBALL COURT
# ============================================================

def draw_court(ax, color="white", lw=1.5):
    """
    Draw a basketball court on a matplotlib axes.
    All measurements are in tenths of a foot (NBA standard).
    """

    # Hoop
    hoop = Circle((0, 0), radius=7.5, linewidth=lw,
                  color=color, fill=False)

    # Backboard
    backboard = Rectangle((-30, -7.5), 60, -1,
                          linewidth=lw, color=color)

    # Paint — outer box
    outer_box = Rectangle((-80, -47.5), 160, 190,
                          linewidth=lw, color=color, fill=False)

    # Paint — inner box
    inner_box = Rectangle((-60, -47.5), 120, 190,
                          linewidth=lw, color=color, fill=False)

    # Free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120,
                         theta1=0, theta2=180,
                         linewidth=lw, color=color)

    # Free throw bottom arc (dashed)
    bottom_free_throw = Arc((0, 142.5), 120, 120,
                            theta1=180, theta2=0,
                            linewidth=lw, color=color,
                            linestyle="dashed")

    # Restricted area
    restricted = Arc((0, 0), 80, 80,
                     theta1=0, theta2=180,
                     linewidth=lw, color=color)

    # Three point line — corner threes
    corner_three_a = Rectangle((-220, -47.5), 0, 140,
                                linewidth=lw, color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140,
                                linewidth=lw, color=color)

    # Three point arc
    three_arc = Arc((0, 0), 475, 475,
                    theta1=22, theta2=158,
                    linewidth=lw, color=color)

    # Center court
    center_outer = Arc((0, 422.5), 120, 120,
                       theta1=180, theta2=0,
                       linewidth=lw, color=color)

    court_elements = [
        hoop, backboard, outer_box, inner_box,
        top_free_throw, bottom_free_throw, restricted,
        corner_three_a, corner_three_b, three_arc,
        center_outer
    ]

    for element in court_elements:
        ax.add_patch(element)

    return ax

# ============================================================
# SHOT ZONE DEFINITIONS
# ============================================================

def get_shot_zone(x, y):
    """
    Classify a shot into a zone based on court location.
    Returns zone name and expected points for that zone.
    """
    # Distance from basket (in tenths of feet)
    dist = np.sqrt(x**2 + y**2)

    # Corner three (x beyond 220, close to baseline)
    if abs(x) >= 220 and y <= 92.5:
        return "Corner Three", 1.09

    # Above the break three
    if dist >= 237:
        return "Above Break Three", 1.05

    # Mid range zones
    if dist >= 120:
        if y <= 92.5:
            return "Mid Range (Baseline)", 0.79
        elif abs(x) <= 80:
            return "Mid Range (Paint Touch)", 0.81
        else:
            return "Mid Range (Wing)", 0.78

    # Paint — close range
    if dist <= 60:
        return "At Rim", 1.28

    # Paint — not at rim
    return "Short Mid Range", 0.76

# Expected points by zone
# Based on NBA averages from tracking data
ZONE_EXPECTED = {
    "At Rim":               1.28,
    "Short Mid Range":      0.76,
    "Mid Range (Baseline)": 0.79,
    "Mid Range (Wing)":     0.78,
    "Mid Range (Paint Touch)": 0.81,
    "Corner Three":         1.09,
    "Above Break Three":    1.05,
}

# ============================================================
# FETCH SHOT DATA
# ============================================================

def get_player_id(name):
    all_players = players.get_players()
    for p in all_players:
        if p["full_name"].lower() == name.lower():
            return p["id"]
    return None

def get_shot_data(player_name, season="2025-26"):
    """Fetch shot chart data for a player."""
    player_id = get_player_id(player_name)
    if not player_id:
        print(f"Player '{player_name}' not found.")
        return None

    print(f"Fetching shot data for {player_name}...")
    time.sleep(0.5)

    shot_chart = shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=player_id,
        season_nullable=season,
        season_type_all_star="Regular Season",
        context_measure_simple="FGA"
    )

    df = shot_chart.get_data_frames()[0]
    if df.empty:
        print(f"No shot data found for {player_name}.")
        return None

    print(f"Found {len(df)} shot attempts.\n")
    return df

# ============================================================
# ANALYZE SHOT QUALITY
# ============================================================

def analyze_shots(df, player_name):
    """Calculate shot quality metrics for a player."""

    # Classify each shot into a zone
    df = df.copy()
    df["ZONE"] = df.apply(
        lambda row: get_shot_zone(row["LOC_X"], row["LOC_Y"])[0],
        axis=1
    )
    df["EXPECTED_PTS"] = df["ZONE"].map(ZONE_EXPECTED)
    df["MADE"] = (df["SHOT_MADE_FLAG"] == 1).astype(int)
    df["POINTS"] = df["MADE"] * df["SHOT_TYPE"].apply(
        lambda x: 3 if "3PT" in str(x) else 2
    )

    # Zone summary
    zone_summary = df.groupby("ZONE").agg(
        attempts=("MADE", "count"),
        makes=("MADE", "sum"),
        actual_pts=("POINTS", "sum"),
        expected_pts_per=("EXPECTED_PTS", "mean")
    ).reset_index()

    zone_summary["fg_pct"] = (
        zone_summary["makes"] / zone_summary["attempts"]
    ).round(3)
    zone_summary["actual_pts_per"] = (
        zone_summary["actual_pts"] / zone_summary["attempts"]
    ).round(3)
    zone_summary["expected_total"] = (
        zone_summary["attempts"] * zone_summary["expected_pts_per"]
    ).round(1)
    zone_summary["pts_above_expected"] = (
        zone_summary["actual_pts"] - zone_summary["expected_total"]
    ).round(1)

    # Overall metrics
    total_attempts = len(df)
    total_actual   = df["POINTS"].sum()
    total_expected = (df["EXPECTED_PTS"]).sum()
    pts_above_exp  = total_actual - total_expected

    print("=" * 60)
    print(f"SHOT QUALITY REPORT — {player_name}")
    print("=" * 60)
    print(f"Total attempts:        {total_attempts}")
    print(f"Actual points:         {total_actual}")
    print(f"Expected points:       {total_expected:.1f}")
    print(f"Points above expected: {pts_above_exp:+.1f}")
    print(f"Pts per attempt:       {total_actual/total_attempts:.3f}")
    print(f"xPts per attempt:      {total_expected/total_attempts:.3f}")
    print()
    print(zone_summary[[
        "ZONE", "attempts", "fg_pct",
        "actual_pts_per", "expected_pts_per",
        "pts_above_expected"
    ]].sort_values("attempts", ascending=False).to_string(index=False))

    return df, zone_summary, pts_above_exp

# ============================================================
# VISUALIZE — COURT HEATMAP
# ============================================================

def plot_shot_chart(df, zone_summary, player_name, pts_above_exp):
    """Create a shot chart visualization on a basketball court."""

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        f"Shot Quality Analysis — {player_name} | 2025-26\n"
        f"Points Above Expected: {pts_above_exp:+.1f}",
        fontsize=14, fontweight="bold", color="white"
    )

    # ── Panel 1: Shot Chart Heatmap ─────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#1a1a2e")

    # Draw court
    draw_court(ax1, color="white", lw=1.5)

    # Color shots by made/missed
    made   = df[df["SHOT_MADE_FLAG"] == 1]
    missed = df[df["SHOT_MADE_FLAG"] == 0]

    ax1.scatter(missed["LOC_X"], missed["LOC_Y"],
                c="salmon", s=8, alpha=0.4, label="Missed")
    ax1.scatter(made["LOC_X"], made["LOC_Y"],
                c="gold", s=8, alpha=0.6, label="Made")

    ax1.set_xlim(-250, 250)
    ax1.set_ylim(-47.5, 470)
    ax1.set_aspect("equal")
    ax1.set_title("Shot Chart — Gold=Made, Red=Missed",
                  color="white", fontsize=11)
    ax1.axis("off")
    ax1.legend(facecolor="#161b22", labelcolor="white",
               loc="upper right", fontsize=9)

    # ── Panel 2: Zone Analysis ──────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")

    zones = zone_summary.sort_values("pts_above_expected")
    colors = ["steelblue" if v >= 0 else "salmon"
              for v in zones["pts_above_expected"]]

    bars = ax2.barh(zones["ZONE"], zones["pts_above_expected"],
                    color=colors, alpha=0.85)

    for bar, val in zip(bars, zones["pts_above_expected"]):
        ax2.text(
            val + (0.5 if val >= 0 else -0.5),
            bar.get_y() + bar.get_height()/2,
            f"{val:+.1f}",
            va="center",
            ha="left" if val >= 0 else "right",
            color="white", fontsize=9
        )

    ax2.axvline(x=0, color="white", linewidth=1, alpha=0.5)
    ax2.set_title(
        "Points Above Expected by Zone\n"
        "Blue = Outperforming | Red = Underperforming",
        color="white", fontsize=11
    )
    ax2.set_xlabel("Points Above Expected", color="white")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333366")

    # Add zone attempt counts as annotations
    for i, row in zones.iterrows():
        ax2.text(
            zones["pts_above_expected"].min() - 1,
            list(zones["ZONE"]).index(row["ZONE"]),
            f"n={int(row['attempts'])}",
            va="center", ha="right",
            color="gray", fontsize=8
        )

    plt.tight_layout()
    filename = f"shots_{player_name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    print(f"\nSaved as {filename}")
    plt.show()

# ============================================================
# COMPARE MULTIPLE PLAYERS
# ============================================================

def compare_shot_quality(player_names, season="2025-26"):
    """Compare shot quality across multiple players."""

    results = []
    for name in player_names:
        df = get_shot_data(name, season)
        if df is None:
            continue
        time.sleep(0.3)

        df["ZONE"] = df.apply(
            lambda row: get_shot_zone(row["LOC_X"], row["LOC_Y"])[0],
            axis=1
        )
        df["EXPECTED_PTS"] = df["ZONE"].map(ZONE_EXPECTED)
        df["MADE"] = (df["SHOT_MADE_FLAG"] == 1).astype(int)
        df["POINTS"] = df["MADE"] * df["SHOT_TYPE"].apply(
            lambda x: 3 if "3PT" in str(x) else 2
        )

        total_attempts = len(df)
        total_actual   = df["POINTS"].sum()
        total_expected = df["EXPECTED_PTS"].sum()

        results.append({
            "Player":           name,
            "Attempts":         total_attempts,
            "Actual Pts":       total_actual,
            "Expected Pts":     round(total_expected, 1),
            "Pts Above Exp":    round(total_actual - total_expected, 1),
            "Pts/Attempt":      round(total_actual / total_attempts, 3),
            "xPts/Attempt":     round(total_expected / total_attempts, 3),
        })

    df_results = pd.DataFrame(results).sort_values(
        "Pts Above Exp", ascending=False
    )

    print("\n" + "=" * 75)
    print("SHOT QUALITY COMPARISON")
    print("=" * 75)
    print(df_results.to_string(index=False))

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        "Shot Quality Comparison — 2025-26 Season",
        fontsize=14, fontweight="bold", color="white"
    )

    for ax in [ax1, ax2]:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333366")

    # Panel 1 — Points above expected
    colors = ["gold" if v > 0 else "salmon"
              for v in df_results["Pts Above Exp"]]
    ax1.barh(df_results["Player"], df_results["Pts Above Exp"],
             color=colors, alpha=0.85)
    ax1.axvline(x=0, color="white", linewidth=1, alpha=0.5)
    ax1.set_title("Total Points Above Expected\n"
                  "Gold = Outperforming xPts",
                  color="white", fontsize=11)
    ax1.set_xlabel("Points Above Expected", color="white")
    ax1.invert_yaxis()

    # Panel 2 — Actual vs Expected per attempt
    x = np.arange(len(df_results))
    width = 0.35
    ax2.bar(x - width/2, df_results["xPts/Attempt"],
            width, label="Expected", color="steelblue", alpha=0.85)
    ax2.bar(x + width/2, df_results["Pts/Attempt"],
            width, label="Actual", color="gold", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [n.split()[-1] for n in df_results["Player"]],
        color="white", fontsize=9
    )
    ax2.set_title("Actual vs Expected Points per Attempt",
                  color="white", fontsize=11)
    ax2.set_ylabel("Points per Attempt", color="white")
    ax2.legend(facecolor="#161b22", labelcolor="white")

    plt.tight_layout()
    plt.savefig("shot_comparison.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    print("\nSaved as shot_comparison.png")
    plt.show()

    return df_results

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Single player analysis
    player_name = "Nikola Jokic"
    df_shots = get_shot_data(player_name)

    if df_shots is not None:
        df_shots, zone_summary, pts_above = analyze_shots(
            df_shots, player_name
        )
        plot_shot_chart(df_shots, zone_summary,
                        player_name, pts_above)

    # Multi-player comparison
    print("\n" + "=" * 40)
    print("Running multi-player comparison...")
    print("=" * 40 + "\n")

    players_to_compare = [
        "Nikola Jokić",
        "Shai Gilgeous-Alexander",
        "Luka Dončić",
        "Stephen Curry",
        "Giannis Antetokounmpo"
    ]

    time.sleep(1)
    compare_shot_quality(players_to_compare)
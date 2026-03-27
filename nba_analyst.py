"""
NBA Analytics Toolkit v1.0
Built by Atakan Karaçoban

A professional NBA analysis tool combining:
- Scouting dashboards
- Player comparisons
- Prospect evaluation
- Correlation analysis
- Regression modeling
- Game score tracking
"""

from nba_api.stats.endpoints import (
    playergamelog,
    playercareerstats,
    leaguedashplayerstats,
    leaguedashteamstats
)
from nba_api.stats.static import players
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import time
import os

# ============================================================
# CONSTANTS — things that never change
# ============================================================
CURRENT_SEASON = "2025-26"
LAST_SEASON = "2024-25"
MIN_GAMES = 20
MIN_MINUTES = 15
YOUNG_PLAYER_AGE = 23

WEIGHTS = {
    "PTS": 1.0,
    "REB": 1.2,
    "AST": 1.5,
    "STL": 2.0,
    "BLK": 1.3,
    "TOV": -2.0
}

# ============================================================
# THE MAIN CLASS
# ============================================================

class NBAAnalyst:
    """
    A professional NBA analytics tool.
    Create one instance and use all its methods.
    Data is cached to avoid redundant API calls.
    """

    def __init__(self):
        """Initialize the analyst with empty caches."""
        self.player_cache = {}       # stores player game logs
        self.career_cache = {}       # stores career stats
        self.league_cache = {}       # stores league-wide stats
        self.player_id_cache = {}    # stores player IDs
        print("NBA Analyst initialized.")

    # ── Private helper methods (start with _) ───────────────
    # Private means "internal use only — don't call from outside"

    def _get_player_id(self, name):
        """Find a player's NBA ID. Cached after first lookup."""
        if name in self.player_id_cache:
            return self.player_id_cache[name]

        all_players = players.get_players()
        for p in all_players:
            if p["full_name"].lower() == name.lower():
                self.player_id_cache[name] = p["id"]
                return p["id"]
        return None

    def _get_gamelog(self, name, season=CURRENT_SEASON):
        """Get a player's game log. Cached after first fetch."""
        cache_key = f"{name}_{season}"
        if cache_key in self.player_cache:
            return self.player_cache[cache_key]

        player_id = self._get_player_id(name)
        if not player_id:
            print(f"Player '{name}' not found.")
            return None

        time.sleep(0.5)
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season
        )
        df = gamelog.get_data_frames()[0]
        df = df.sort_values("GAME_DATE").reset_index(drop=True)
        self.player_cache[cache_key] = df
        return df

    def _get_career(self, name):
        """Get a player's career stats. Cached after first fetch."""
        if name in self.career_cache:
            return self.career_cache[name]

        player_id = self._get_player_id(name)
        if not player_id:
            return None

        time.sleep(0.5)
        career = playercareerstats.PlayerCareerStats(
            player_id=player_id
        )
        df = career.get_data_frames()[0]
        df = df[df["GP"] >= MIN_GAMES].copy()
        df["FG_PCT"] = df["FGM"] / df["FGA"]
        df["PPG"] = (df["PTS"] / df["GP"]).round(1)
        df["RPG"] = (df["REB"] / df["GP"]).round(1)
        df["APG"] = (df["AST"] / df["GP"]).round(1)
        df["SEASON_YEAR"] = df["SEASON_ID"].str[:4].astype(int)
        self.career_cache[name] = df
        return df

    def _get_league_stats(self, season=CURRENT_SEASON):
        """Get league-wide stats. Cached after first fetch."""
        if season in self.league_cache:
            return self.league_cache[season]

        time.sleep(0.5)
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame"
        ).get_data_frames()[0]
        df = df[df["GP"] >= MIN_GAMES].copy()
        self.league_cache[season] = df
        return df

    def _calculate_game_score(self, row):
        """Calculate Game Score for a single game row."""
        return round(
            row["PTS"] + (0.4 * row["FGM"])
            - (0.7 * row["FGA"])
            - (0.4 * (row["FTA"] - row["FTM"]))
            + (0.7 * row["OREB"]) + (0.3 * row["DREB"])
            + row["STL"] + (0.7 * row["AST"])
            + (0.7 * row["BLK"]) - (0.4 * row["PF"])
            - row["TOV"], 2
        )

    def _get_percentile(self, value, series):
        """Calculate percentile of a value within a series."""
        return round(stats.percentileofscore(
            series.dropna(), value
        ), 1)

    def _weighted_score(self, df, prefix=""):
        """Calculate weighted efficiency score for a DataFrame."""
        pts = df[f"{prefix}PTS"]
        reb = df[f"{prefix}REB"]
        ast = df[f"{prefix}AST"]
        stl = df[f"{prefix}STL"]
        blk = df[f"{prefix}BLK"]
        tov = df[f"{prefix}TOV"]
        fgp = df[f"{prefix}FG_PCT"] if f"{prefix}FG_PCT" in df.columns else df["FG_PCT"]

        return (
            pts  * WEIGHTS["PTS"] +
            reb  * WEIGHTS["REB"] +
            ast  * WEIGHTS["AST"] +
            stl  * WEIGHTS["STL"] +
            blk  * WEIGHTS["BLK"] +
            tov  * WEIGHTS["TOV"]
        ) * fgp

    def _save_figure(self, filename):
        """Save current figure and show it."""
        plt.savefig(filename, dpi=150,
                    bbox_inches="tight", facecolor="#0d1117")
        print(f"Saved as {filename}")
        plt.show()

    # ── Public methods — the actual tools ───────────────────

    def game_score_tracker(self, player_name):
        """Track a player's Game Score across the current season."""
        print(f"\nGenerating Game Score tracker for {player_name}...")
        df = self._get_gamelog(player_name)
        if df is None:
            return

        df["GAME_SCORE"] = df.apply(
            self._calculate_game_score, axis=1
        )
        gs_avg = df["GAME_SCORE"].mean()
        gs_max = df["GAME_SCORE"].max()

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")

        colors = ["gold" if g >= 30 else "steelblue" if g >= 20
                  else "cadetblue" if g >= 10 else "salmon"
                  for g in df["GAME_SCORE"]]

        ax.bar(range(len(df)), df["GAME_SCORE"],
               color=colors, alpha=0.85)
        ax.axhline(y=gs_avg, color="white", linestyle="--",
                   linewidth=1.5, label=f"Avg: {gs_avg:.1f}")

        ax.set_title(
            f"{player_name} — Game Score Tracker {CURRENT_SEASON}\n"
            f"Average: {gs_avg:.1f} | Peak: {gs_max:.1f}",
            color="white", fontsize=13
        )
        ax.set_xlabel("Game Number", color="white")
        ax.set_ylabel("Game Score", color="white")
        ax.legend(facecolor="#161b22", labelcolor="white")
        ax.tick_params(colors="white")
        ax.text(0.01, 0.95,
                "🟡 ≥30 Historic  🔵 ≥20 Great  "
                "🩵 ≥10 Solid  🔴 <10 Poor",
                transform=ax.transAxes, fontsize=9,
                verticalalignment="top", color="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333366")

        plt.tight_layout()
        filename = f"gs_{player_name.replace(' ', '_')}.png"
        self._save_figure(filename)

    def prospect_rankings(self):
        """Rank current NBA prospects by age-adjusted score."""
        print("\nGenerating prospect rankings...")

        df_curr = self._get_league_stats(CURRENT_SEASON)
        df_last = self._get_league_stats(LAST_SEASON)

        df_curr = df_curr[df_curr["AGE"] <= YOUNG_PLAYER_AGE].copy()
        df_curr = df_curr[df_curr["MIN"] >= MIN_MINUTES].copy()

        df_curr["AGE_FACTOR"] = df_curr["AGE"].apply(
            lambda age: max(1.0, 1.0 + (23 - age) * 0.1)
        )
        df_curr["BASE_SCORE"] = self._weighted_score(df_curr)
        df_curr["PROSPECT_SCORE"] = (
            df_curr["BASE_SCORE"] * df_curr["AGE_FACTOR"]
        ).round(2)

        df_last["BASE_SCORE_LAST"] = self._weighted_score(df_last)
        df_merged = df_curr.merge(
            df_last[["PLAYER_ID", "BASE_SCORE_LAST"]],
            on="PLAYER_ID", how="left"
        )
        df_merged["IMPROVEMENT"] = (
            df_merged["BASE_SCORE"] - df_merged["BASE_SCORE_LAST"]
        ).round(2).fillna(0)
        df_merged["IS_ROOKIE"] = df_merged["BASE_SCORE_LAST"].isna()
        df_merged["FINAL_SCORE"] = (
            df_merged["PROSPECT_SCORE"] * 0.7 +
            df_merged["IMPROVEMENT"] * 0.3
        ).round(2)

        top20 = df_merged.sort_values(
            "FINAL_SCORE", ascending=False
        ).head(20).reset_index(drop=True)

        print("\n" + "=" * 60)
        print("TOP 20 NBA PROSPECTS")
        print("=" * 60)
        for i, row in top20.iterrows():
            status = ("ROOKIE" if row["IS_ROOKIE"]
                      else f"↑{row['IMPROVEMENT']:.1f}"
                      if row["IMPROVEMENT"] > 0
                      else f"↓{abs(row['IMPROVEMENT']):.1f}")
            print(f"{i+1:>2}. {row['PLAYER_NAME']:<25} "
                  f"Age {int(row['AGE'])}  "
                  f"Score: {row['FINAL_SCORE']:.2f}  {status}")

        fig, ax = plt.subplots(figsize=(14, 10))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")

        colors = ["gold" if s > 15 else "steelblue"
                  for s in top20["FINAL_SCORE"]]
        ax.barh(
            [f"{r['PLAYER_NAME']} ({int(r['AGE'])})"
             for _, r in top20.iterrows()],
            top20["FINAL_SCORE"],
            color=colors, alpha=0.85
        )
        ax.invert_yaxis()
        ax.set_title(
            f"NBA Prospect Rankings — {CURRENT_SEASON}\n"
            "Age-Adjusted | Gold = Elite",
            color="white", fontsize=13
        )
        ax.set_xlabel("Prospect Score", color="white")
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333366")

        plt.tight_layout()
        self._save_figure("prospects.png")

    def correlation_analysis(self):
        """Analyze which stats correlate most with winning."""
        print("\nRunning correlation analysis...")

        time.sleep(0.5)
        df = leaguedashteamstats.LeagueDashTeamStats(
            season=CURRENT_SEASON,
            measure_type_detailed_defense="Base"
        ).get_data_frames()[0]

        df["PPG"]     = df["PTS"] / df["GP"]
        df["RPG"]     = df["REB"] / df["GP"]
        df["APG"]     = df["AST"] / df["GP"]
        df["SPG"]     = df["STL"] / df["GP"]
        df["BPG"]     = df["BLK"] / df["GP"]
        df["TOPG"]    = df["TOV"] / df["GP"]
        df["FG_PCT"]  = df["FGM"] / df["FGA"]
        df["FG3_PCT"] = df["FG3M"] / df["FG3A"]
        df["WIN_PCT"] = df["W_PCT"]

        stat_map = {
            "Points/Game": "PPG",
            "Rebounds/Game": "RPG",
            "Assists/Game": "APG",
            "Turnovers/Game": "TOPG",
            "Steals/Game": "SPG",
            "Blocks/Game": "BPG",
            "FG%": "FG_PCT",
            "3PT%": "FG3_PCT",
        }

        print("\n=== CORRELATION WITH WINNING ===\n")
        results = []
        for name, col in stat_map.items():
            clean = df[[col, "WIN_PCT"]].dropna()
            r, p = stats.pearsonr(clean[col], clean["WIN_PCT"])
            strength = ("STRONG" if abs(r) >= 0.7
                        else "MODERATE" if abs(r) >= 0.4
                        else "WEAK" if abs(r) >= 0.2
                        else "NEGLIGIBLE")
            results.append({
                "Stat": name, "r": round(r, 3),
                "p": round(p, 4), "Strength": strength,
                "col": col
            })
            print(f"{name:<20} r={r:+.3f}  {strength}")

        results_df = pd.DataFrame(results)
        results_df["abs_r"] = results_df["r"].abs()
        results_df = results_df.sort_values("abs_r")

        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        fig.patch.set_facecolor("#0d1117")
        fig.suptitle(
            f"Which Stats Predict Winning? — {CURRENT_SEASON}",
            fontsize=14, color="white", fontweight="bold"
        )

        for idx, row in enumerate(results_df.itertuples()):
            if idx >= 8:
                break
            ax = axes.flatten()[idx]
            ax.set_facecolor("#161b22")
            clean = df[[row.col, "WIN_PCT"]].dropna()
            sc = plt.cm.RdYlGn(
                (clean["WIN_PCT"] - clean["WIN_PCT"].min()) /
                (clean["WIN_PCT"].max() - clean["WIN_PCT"].min())
            )
            ax.scatter(clean[row.col], clean["WIN_PCT"],
                       c=sc, s=60, alpha=0.8)
            z = np.polyfit(clean[row.col], clean["WIN_PCT"], 1)
            x_l = np.linspace(
                clean[row.col].min(), clean[row.col].max(), 100
            )
            ax.plot(x_l, np.poly1d(z)(x_l), "--",
                    color="yellow", alpha=0.7, linewidth=1.5)
            r_color = ("lightgreen" if abs(row.r) >= 0.4
                       else "salmon")
            ax.set_title(
                f"{row.Stat}\nr={row.r:+.3f} ({row.Strength})",
                color=r_color, fontsize=9
            )
            ax.tick_params(colors="white", labelsize=7)
            ax.grid(color="gray", alpha=0.2)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333366")

        plt.tight_layout()
        self._save_figure("correlation.png")

    def regression_analysis(self):
        """Run regression to find mathematically optimal weights."""
        print("\nRunning regression analysis...")
        print("Fetching 5 seasons of data...\n")

        seasons = ["2021-22", "2022-23", "2023-24",
                   "2024-25", "2025-26"]
        dfs = []
        for season in seasons:
            print(f"  Fetching {season}...")
            time.sleep(0.5)
            df = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Base"
            ).get_data_frames()[0]
            df["SEASON"] = season
            dfs.append(df)

        df_all = pd.concat(dfs, ignore_index=True)
        df_all["PPG"]     = df_all["PTS"]  / df_all["GP"]
        df_all["RPG"]     = df_all["REB"]  / df_all["GP"]
        df_all["APG"]     = df_all["AST"]  / df_all["GP"]
        df_all["SPG"]     = df_all["STL"]  / df_all["GP"]
        df_all["BPG"]     = df_all["BLK"]  / df_all["GP"]
        df_all["TOPG"]    = df_all["TOV"]  / df_all["GP"]
        df_all["FG_PCT"]  = df_all["FGM"]  / df_all["FGA"]
        df_all["FG3_PCT"] = df_all["FG3M"] / df_all["FG3A"]
        df_all["FT_PCT"]  = df_all["FTM"]  / df_all["FTA"]
        df_all["WIN_PCT"] = df_all["W_PCT"]

        features = ["PPG", "RPG", "APG", "SPG", "BPG",
                    "TOPG", "FG_PCT", "FG3_PCT", "FT_PCT"]
        df_clean = df_all[features + ["WIN_PCT"]].dropna()

        X = df_clean[features].values
        y = df_clean["WIN_PCT"].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)

        print(f"\nR² Score: {r2:.4f}")
        print(f"Model explains {r2*100:.1f}% of win% variation\n")
        print(f"{'Stat':<12} {'Coefficient':>14} {'Direction':>12}")
        print("─" * 42)

        coefs = pd.DataFrame({
            "Stat": features,
            "Coef": model.coef_
        }).sort_values("Coef", key=abs, ascending=False)

        for _, row in coefs.iterrows():
            direction = "↑ Helps" if row["Coef"] > 0 else "↓ Hurts"
            print(f"{row['Stat']:<12} {row['Coef']:>+14.4f} "
                  f"{direction:>12}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor("#0d1117")
        fig.suptitle(
            f"Regression Analysis — {len(df_clean)} Team-Seasons | "
            f"R²={r2:.3f}",
            fontsize=14, color="white", fontweight="bold"
        )

        for ax in [ax1, ax2]:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333366")

        colors = ["steelblue" if c > 0 else "salmon"
                  for c in coefs["Coef"]]
        ax1.barh(coefs["Stat"], coefs["Coef"],
                 color=colors, alpha=0.85)
        ax1.axvline(x=0, color="white", linewidth=1, alpha=0.5)
        ax1.set_title("Regression Coefficients\n"
                      "Blue=Helps | Red=Hurts",
                      color="white", fontsize=11)
        ax1.set_xlabel("Coefficient", color="white")

        sc = plt.cm.RdYlGn(
            (y - y.min()) / (y.max() - y.min())
        )
        ax2.scatter(y, y_pred, c=sc, s=50, alpha=0.7)
        ax2.plot([y.min(), y.max()], [y.min(), y.max()],
                 "--", color="yellow", linewidth=1.5,
                 label="Perfect prediction")
        ax2.set_title("Predicted vs Actual Win%",
                      color="white", fontsize=11)
        ax2.set_xlabel("Actual Win%", color="white")
        ax2.set_ylabel("Predicted Win%", color="white")
        ax2.legend(facecolor="#161b22", labelcolor="white")
        ax2.grid(color="gray", alpha=0.2)

        plt.tight_layout()
        self._save_figure("regression.png")

# ============================================================
# MAIN MENU — runs when you execute the script
# ============================================================

def print_header():
    print("\n" + "═" * 45)
    print("║     NBA ANALYTICS TOOLKIT v1.0          ║")
    print("║     Built by Atakan Karaçoban           ║")
    print("═" * 45)

def main():
    print_header()
    analyst = NBAAnalyst()  # create ONE instance, reuse it

    while True:
        print("\n" + "─" * 45)
        print("  1. Game Score Tracker")
        print("  2. Prospect Rankings")
        print("  3. Correlation Analysis")
        print("  4. Regression Analysis")
        print("  5. Exit")
        print("─" * 45)

        choice = input("\nSelect option (1-5): ").strip()

        if choice == "1":
            name = input("Enter player name: ").strip()
            analyst.game_score_tracker(name)

        elif choice == "2":
            analyst.prospect_rankings()

        elif choice == "3":
            analyst.correlation_analysis()

        elif choice == "4":
            analyst.regression_analysis()

        elif choice == "5":
            print("\nGoodbye.\n")
            break

        else:
            print("Invalid option. Choose 1-5.")

if __name__ == "__main__":
    main()
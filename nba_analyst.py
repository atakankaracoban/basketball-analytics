"""
NBA Analytics Toolkit v1.0
Built by Atakan Karaçoban
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
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import time

# ============================================================
# CONSTANTS
# ============================================================
CURRENT_SEASON = "2025-26"
LAST_SEASON    = "2024-25"
MIN_GAMES      = 20
MIN_MINUTES    = 15
YOUNG_AGE      = 23

WEIGHTS_INTUITIVE = {
    "PTS": 1.0,
    "REB": 1.5,
    "AST": 1.5,
    "STL": 2.0,
    "BLK": 1.0,
    "TOV": -2.0
}

WEIGHTS_REGRESSION = {
    "PTS": 0.0,
    "REB": 1.7,
    "AST": 0.5,
    "STL": 1.3,
    "BLK": 0.2,
    "TOV": -1.2
}

# ============================================================
# NBA ANALYST CLASS
# ============================================================

class NBAAnalyst:

    def __init__(self):
        self.player_cache    = {}
        self.career_cache    = {}
        self.league_cache    = {}
        self.player_id_cache = {}
        self.weights         = WEIGHTS_INTUITIVE
        print("NBA Analyst initialized.")
        print("Weight system: Adjusted Intuition\n")

    # ── Private helpers ─────────────────────────────────────

    def _get_player_id(self, name):
        if name in self.player_id_cache:
            return self.player_id_cache[name]
        for p in players.get_players():
            if p["full_name"].lower() == name.lower():
                self.player_id_cache[name] = p["id"]
                return p["id"]
        return None

    def _get_gamelog(self, name, season=None):
        if season is None:
            season = CURRENT_SEASON
        key = f"{name}_{season}"
        if key in self.player_cache:
            return self.player_cache[key]
        pid = self._get_player_id(name)
        if not pid:
            print(f"Player '{name}' not found.")
            return None
        time.sleep(0.5)
        df = playergamelog.PlayerGameLog(
            player_id=pid, season=season
        ).get_data_frames()[0]
        df = df.sort_values("GAME_DATE").reset_index(drop=True)
        self.player_cache[key] = df
        return df

    def _get_career(self, name):
        if name in self.career_cache:
            return self.career_cache[name]
        pid = self._get_player_id(name)
        if not pid:
            return None
        time.sleep(0.5)
        df = playercareerstats.PlayerCareerStats(
            player_id=pid
        ).get_data_frames()[0]
        df = df[df["GP"] >= MIN_GAMES].copy()
        df["FG_PCT"]     = df["FGM"] / df["FGA"]
        df["PPG"]        = (df["PTS"] / df["GP"]).round(1)
        df["RPG"]        = (df["REB"] / df["GP"]).round(1)
        df["APG"]        = (df["AST"] / df["GP"]).round(1)
        df["SEASON_YEAR"]= df["SEASON_ID"].str[:4].astype(int)
        self.career_cache[name] = df
        return df

    def _get_league(self, season=None):
        if season is None:
            season = CURRENT_SEASON
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

    def _game_score(self, row):
        return round(
            row["PTS"] + (0.4 * row["FGM"])
            - (0.7 * row["FGA"])
            - (0.4 * (row["FTA"] - row["FTM"]))
            + (0.7 * row["OREB"]) + (0.3 * row["DREB"])
            + row["STL"] + (0.7 * row["AST"])
            + (0.7 * row["BLK"]) - (0.4 * row["PF"])
            - row["TOV"], 2
        )

    def _percentile(self, value, series):
        return round(
            stats.percentileofscore(series.dropna(), value), 1
        )

    def _wscore(self, df):
        w = self.weights
        fg = df["FG_PCT"] if "FG_PCT" in df.columns else (
            df["FGM"] / df["FGA"]
        )
        return (
            df["PTS"] * w["PTS"] +
            df["REB"] * w["REB"] +
            df["AST"] * w["AST"] +
            df["STL"] * w["STL"] +
            df["BLK"] * w["BLK"] +
            df["TOV"] * w["TOV"]
        ) * fg

    def _style(self, ax):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333366")

    def _save(self, filename):
        plt.savefig(filename, dpi=150,
                    bbox_inches="tight", facecolor="#0d1117")
        print(f"Saved as {filename}")
        plt.show()

    # ── Public tools ─────────────────────────────────────────

    def game_score_tracker(self, name):
        print(f"\nGame Score tracker: {name}")
        df = self._get_gamelog(name)
        if df is None:
            return
        df["GS"] = df.apply(self._game_score, axis=1)
        avg = df["GS"].mean()
        peak = df["GS"].max()

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor("#0d1117")
        self._style(ax)

        colors = [
            "gold" if g >= 30 else
            "steelblue" if g >= 20 else
            "cadetblue" if g >= 10 else
            "salmon" for g in df["GS"]
        ]
        ax.bar(range(len(df)), df["GS"], color=colors, alpha=0.85)
        ax.axhline(y=avg, color="white", linestyle="--",
                   linewidth=1.5, label=f"Avg: {avg:.1f}")
        ax.set_title(
            f"{name} — Game Score Tracker {CURRENT_SEASON}\n"
            f"Average: {avg:.1f}  |  Peak: {peak:.1f}",
            color="white", fontsize=13
        )
        ax.set_xlabel("Game Number", color="white")
        ax.set_ylabel("Game Score", color="white")
        ax.legend(facecolor="#161b22", labelcolor="white")
        ax.text(0.01, 0.95,
                "🟡 ≥30 Historic  🔵 ≥20 Great  "
                "🩵 ≥10 Solid  🔴 <10 Poor",
                transform=ax.transAxes, fontsize=9,
                verticalalignment="top", color="white")
        plt.tight_layout()
        self._save(f"gs_{name.replace(' ', '_')}.png")

    def prospect_rankings(self):
        print("\nGenerating prospect rankings...")
        w_name = ("Adjusted Intuition"
                  if self.weights == WEIGHTS_INTUITIVE
                  else "Data-Driven Regression")
        print(f"Using weight system: {w_name}\n")

        df_curr = self._get_league(CURRENT_SEASON).copy()
        df_last = self._get_league(LAST_SEASON).copy()

        df_curr = df_curr[df_curr["AGE"] <= YOUNG_AGE].copy()
        df_curr = df_curr[df_curr["MIN"] >= MIN_MINUTES].copy()

        df_curr["AGE_FACTOR"] = df_curr["AGE"].apply(
            lambda a: max(1.0, 1.0 + (23 - a) * 0.1)
        )
        df_curr["BASE"]      = self._wscore(df_curr)
        df_curr["PROSPECT"]  = (
            df_curr["BASE"] * df_curr["AGE_FACTOR"]
        ).round(2)

        df_last["BASE_LAST"] = self._wscore(df_last)
        df = df_curr.merge(
            df_last[["PLAYER_ID", "BASE_LAST"]],
            on="PLAYER_ID", how="left"
        )
        df["IMPROVE"]  = (df["BASE"] - df["BASE_LAST"]).round(2).fillna(0)
        df["IS_ROOKIE"]= df["BASE_LAST"].isna()
        df["FINAL"]    = (
            df["PROSPECT"] * 0.7 + df["IMPROVE"] * 0.3
        ).round(2)

        top20 = df.sort_values(
            "FINAL", ascending=False
        ).head(20).reset_index(drop=True)

        print("=" * 55)
        print(f"TOP 20 PROSPECTS — {w_name}")
        print("=" * 55)
        for i, row in top20.iterrows():
            tag = ("ROOKIE" if row["IS_ROOKIE"]
                   else f"↑{row['IMPROVE']:.1f}"
                   if row["IMPROVE"] > 0
                   else f"↓{abs(row['IMPROVE']):.1f}")
            print(f"{i+1:>2}. {row['PLAYER_NAME']:<25} "
                  f"Age {int(row['AGE'])}  "
                  f"Score: {row['FINAL']:.2f}  {tag}")

        fig, ax = plt.subplots(figsize=(14, 10))
        fig.patch.set_facecolor("#0d1117")
        self._style(ax)

        colors = ["gold" if s > 15 else "steelblue"
                  for s in top20["FINAL"]]
        ax.barh(
            [f"{r['PLAYER_NAME']} ({int(r['AGE'])})"
             for _, r in top20.iterrows()],
            top20["FINAL"], color=colors, alpha=0.85
        )
        ax.invert_yaxis()
        ax.set_title(
            f"NBA Prospect Rankings — {CURRENT_SEASON}\n"
            f"Weight System: {w_name}",
            color="white", fontsize=13
        )
        ax.set_xlabel("Prospect Score", color="white")
        ax.tick_params(colors="white", labelsize=9)
        plt.tight_layout()
        self._save("prospects.png")

    def correlation_analysis(self):
        print("\nRunning correlation analysis...")
        time.sleep(0.5)
        df = leaguedashteamstats.LeagueDashTeamStats(
            season=CURRENT_SEASON,
            measure_type_detailed_defense="Base"
        ).get_data_frames()[0]

        df["PPG"]    = df["PTS"]  / df["GP"]
        df["RPG"]    = df["REB"]  / df["GP"]
        df["APG"]    = df["AST"]  / df["GP"]
        df["SPG"]    = df["STL"]  / df["GP"]
        df["BPG"]    = df["BLK"]  / df["GP"]
        df["TOPG"]   = df["TOV"]  / df["GP"]
        df["FG_PCT"] = df["FGM"]  / df["FGA"]
        df["FG3_PCT"]= df["FG3M"] / df["FG3A"]
        df["WIN_PCT"]= df["W_PCT"]

        stat_map = {
            "Points/Game":    "PPG",
            "Rebounds/Game":  "RPG",
            "Assists/Game":   "APG",
            "Turnovers/Game": "TOPG",
            "Steals/Game":    "SPG",
            "Blocks/Game":    "BPG",
            "FG%":            "FG_PCT",
            "3PT%":           "FG3_PCT",
        }

        results = []
        print("\n=== CORRELATION WITH WINNING ===\n")
        for name, col in stat_map.items():
            clean = df[[col, "WIN_PCT"]].dropna()
            r, p = stats.pearsonr(clean[col], clean["WIN_PCT"])
            strength = (
                "STRONG"     if abs(r) >= 0.7 else
                "MODERATE"   if abs(r) >= 0.4 else
                "WEAK"       if abs(r) >= 0.2 else
                "NEGLIGIBLE"
            )
            results.append({
                "Stat": name, "r": round(r, 3),
                "Strength": strength, "col": col
            })
            print(f"{name:<20} r={r:+.3f}  {strength}")

        rdf = pd.DataFrame(results)
        rdf["abs_r"] = rdf["r"].abs()
        rdf = rdf.sort_values("abs_r")

        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        fig.patch.set_facecolor("#0d1117")
        fig.suptitle(
            f"Which Stats Predict Winning? — {CURRENT_SEASON}",
            fontsize=14, color="white", fontweight="bold"
        )
        for idx, row in enumerate(rdf.itertuples()):
            if idx >= 8:
                break
            ax = axes.flatten()[idx]
            self._style(ax)
            clean = df[[row.col, "WIN_PCT"]].dropna()
            sc = plt.cm.RdYlGn(
                (clean["WIN_PCT"] - clean["WIN_PCT"].min()) /
                (clean["WIN_PCT"].max() - clean["WIN_PCT"].min())
            )
            ax.scatter(clean[row.col], clean["WIN_PCT"],
                       c=sc, s=60, alpha=0.8)
            z = np.polyfit(clean[row.col], clean["WIN_PCT"], 1)
            xl = np.linspace(
                clean[row.col].min(), clean[row.col].max(), 100
            )
            ax.plot(xl, np.poly1d(z)(xl), "--",
                    color="yellow", alpha=0.7, linewidth=1.5)
            rc = "lightgreen" if abs(row.r) >= 0.4 else "salmon"
            ax.set_title(
                f"{row.Stat}\nr={row.r:+.3f} ({row.Strength})",
                color=rc, fontsize=9
            )
            ax.tick_params(colors="white", labelsize=7)
            ax.grid(color="gray", alpha=0.2)
        plt.tight_layout()
        self._save("correlation.png")

    def regression_analysis(self):
        print("\nRunning regression analysis...")
        seasons = ["2021-22", "2022-23", "2023-24",
                   "2024-25", "2025-26"]
        dfs = []
        for s in seasons:
            print(f"  Fetching {s}...")
            time.sleep(0.5)
            df = leaguedashteamstats.LeagueDashTeamStats(
                season=s,
                measure_type_detailed_defense="Base"
            ).get_data_frames()[0]
            df["SEASON"] = s
            dfs.append(df)

        df_all = pd.concat(dfs, ignore_index=True)
        df_all["PPG"]    = df_all["PTS"]  / df_all["GP"]
        df_all["RPG"]    = df_all["REB"]  / df_all["GP"]
        df_all["APG"]    = df_all["AST"]  / df_all["GP"]
        df_all["SPG"]    = df_all["STL"]  / df_all["GP"]
        df_all["BPG"]    = df_all["BLK"]  / df_all["GP"]
        df_all["TOPG"]   = df_all["TOV"]  / df_all["GP"]
        df_all["FG_PCT"] = df_all["FGM"]  / df_all["FGA"]
        df_all["FG3_PCT"]= df_all["FG3M"] / df_all["FG3A"]
        df_all["FT_PCT"] = df_all["FTM"]  / df_all["FTA"]
        df_all["WIN_PCT"]= df_all["W_PCT"]

        feats = ["PPG","RPG","APG","SPG","BPG",
                 "TOPG","FG_PCT","FG3_PCT","FT_PCT"]
        dfc = df_all[feats + ["WIN_PCT"]].dropna()
        X = dfc[feats].values
        y = dfc["WIN_PCT"].values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(Xs, y)
        yp = model.predict(Xs)
        r2 = r2_score(y, yp)

        print(f"\nR² = {r2:.4f} ({r2*100:.1f}% of variance explained)\n")
        coefs = pd.DataFrame({
            "Stat": feats, "Coef": model.coef_
        }).sort_values("Coef", key=abs, ascending=False)

        for _, row in coefs.iterrows():
            d = "↑ Helps" if row["Coef"] > 0 else "↓ Hurts"
            print(f"{row['Stat']:<12} {row['Coef']:>+.4f}  {d}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor("#0d1117")
        fig.suptitle(
            f"Regression — {len(dfc)} Team-Seasons | R²={r2:.3f}",
            fontsize=14, color="white", fontweight="bold"
        )
        self._style(ax1)
        self._style(ax2)

        colors = ["steelblue" if c > 0 else "salmon"
                  for c in coefs["Coef"]]
        ax1.barh(coefs["Stat"], coefs["Coef"],
                 color=colors, alpha=0.85)
        ax1.axvline(x=0, color="white", linewidth=1, alpha=0.5)
        ax1.set_title("Regression Coefficients",
                      color="white", fontsize=11)
        ax1.set_xlabel("Coefficient", color="white")

        sc = plt.cm.RdYlGn(
            (y - y.min()) / (y.max() - y.min())
        )
        ax2.scatter(y, yp, c=sc, s=50, alpha=0.7)
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
        self._save("regression.png")

    def switch_weights(self):
        if self.weights == WEIGHTS_INTUITIVE:
            self.weights = WEIGHTS_REGRESSION
            print("\n✓ Switched to: Data-Driven (Regression) weights")
            print("  REB=1.7  AST=0.5  STL=1.3  BLK=0.2  TOV=-1.2")
        else:
            self.weights = WEIGHTS_INTUITIVE
            print("\n✓ Switched to: Adjusted Intuition weights")
            print("  REB=1.5  AST=1.5  STL=2.0  BLK=1.0  TOV=-2.0")

# ============================================================
# MAIN MENU
# ============================================================

def print_header():
    print("\n" + "═" * 45)
    print("  NBA ANALYTICS TOOLKIT v1.0")
    print("  Built by Atakan Karaçoban")
    print("═" * 45)

def main():
    print_header()
    analyst = NBAAnalyst()

    while True:
        w_name = ("Adjusted Intuition"
                  if analyst.weights == WEIGHTS_INTUITIVE
                  else "Data-Driven")
        print("\n" + "─" * 45)
        print(f"  Active weights: {w_name}")
        print("─" * 45)
        print("  1. Game Score Tracker")
        print("  2. Prospect Rankings")
        print("  3. Correlation Analysis")
        print("  4. Regression Analysis")
        print("  5. Switch Weight System")
        print("  6. Exit")
        print("─" * 45)

        choice = input("\nSelect option (1-6): ").strip()

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
            analyst.switch_weights()
        elif choice == "6":
            print("\nGoodbye.\n")
            break
        else:
            print("Invalid option. Choose 1-6.")

if __name__ == "__main__":
    main()
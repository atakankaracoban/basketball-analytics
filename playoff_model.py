"""
NBA Playoff Viability Model — 2025-26 Season
=============================================
Atakan Karaçoban'ın playoff başarı kriterleri üzerine kurulu
ölçülebilir tahmin sistemi.

Kriterler:
  1. Shot Creation Under Constraint  — baskı altında ofansif üretim
  2. Defensive Scalability           — defansif uyum kapasitesi
  3. Weak Link Exposure              — en zayıf halkanın dayanıklılığı
  4. Decision-Making Under Pressure  — baskı altında karar kalitesi
  5. Roster Optionality              — kadro esnekliği

Veri: nba_api (gerçek zamanlı)
Çıktı: 16 takım için playoff viability skoru + bracket tahmini
"""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.patches as mpatches  # type: ignore
import time
import warnings
warnings.filterwarnings("ignore")

from nba_api.stats.endpoints import (  # type: ignore
    leaguedashteamstats,
    leaguedashteamclutch,
    leaguedashlineups,
    teamdashboardbygeneralsplits,
    leaguedashplayerstats,
)

# ============================================================
# PLAYOFF TAKIMLARI — 2025-26
# ============================================================
# Kesinleşmiş + play-in projeksiyonları (12 Nisan sonrası güncelle)

PLAYOFF_TEAMS_EAST = {
    1: "Detroit Pistons",
    2: "Boston Celtics",
    3: "New York Knicks",       # Cavs ile yakın, güncellenebilir
    4: "Cleveland Cavaliers",
    5: "Atlanta Hawks",         # Play-in'den gelecek
    6: "Toronto Raptors",       # Play-in'den gelecek — Atlanta ile yarışta
    7: "Philadelphia 76ers",    # Play-in projeksiyonu
    8: "Charlotte Hornets",     # Play-in projeksiyonu
}

PLAYOFF_TEAMS_WEST = {
    1: "Oklahoma City Thunder",
    2: "San Antonio Spurs",
    3: "Denver Nuggets",
    4: "Houston Rockets",       # Lakers ile yer değiştirebilir
    5: "Los Angeles Lakers",    # Luka + Reaves yaralı — kritik
    6: "Minnesota Timberwolves",
    7: "LA Clippers",  # Play-in projeksiyonu
    8: "Golden State Warriors", # Play-in projeksiyonu
}

# Team ID'leri (nba_api için)
TEAM_IDS = {
    "Detroit Pistons":          1610612765,
    "Boston Celtics":           1610612738,
    "New York Knicks":          1610612752,
    "Cleveland Cavaliers":      1610612739,
    "Atlanta Hawks":            1610612737,
    "Toronto Raptors":          1610612761,
    "Philadelphia 76ers":       1610612755,
    "Charlotte Hornets":        1610612766,
    "Oklahoma City Thunder":    1610612760,
    "San Antonio Spurs":        1610612759,
    "Denver Nuggets":           1610612743,
    "Houston Rockets":          1610612745,
    "Los Angeles Lakers":       1610612747,
    "Minnesota Timberwolves":   1610612750,
    "LA Clippers":              1610612746,
    "Golden State Warriors":    1610612744,
}

SEASON = "2025-26"
# ============================================================
# HEALTH MODIFIER — Manuel yaralanma/sağlık düzeltmesi
# ============================================================
# Aralık: -2.0 (kritik yaralanma) → +0.5 (tam sağlık/momentum)
# 0.0 = nötr, dokunma
# Playoff başlamadan önce güncelle.

HEALTH_MODIFIERS = {
    "Los Angeles Lakers":       -1.5,  # Luka + Reaves out
    "Detroit Pistons":          -0.5,  # Cade son haftalarda sınırlı
    "Boston Celtics":           +0.3,  # Tatum döndü, momentum var
    "Oklahoma City Thunder":     0.0,  # Tam sağlık
    "San Antonio Spurs":         0.0,
    "Denver Nuggets":            0.0,
    "New York Knicks":           0.0,
    "Cleveland Cavaliers":       0.0,
    "Houston Rockets":           0.0,
    "Minnesota Timberwolves":    0.0,
    "LA Clippers":               0.0,
    "Golden State Warriors":     0.0,
    "Atlanta Hawks":             0.0,
    "Toronto Raptors":           0.0,
    "Philadelphia 76ers":        0.0,
    "Charlotte Hornets":         0.0,
}
# ============================================================
# VERİ ÇEKME
# ============================================================

def fetch_base_team_stats():
    """Temel takım istatistikleri — ofans ve defans."""
    print("Fetching base team stats...")
    time.sleep(0.6)

    # Ofansif istatistikler
    df_off = leaguedashteamstats.LeagueDashTeamStats(
        season=SEASON,
        measure_type_detailed_defense="Base",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]

    # Defansif istatistikler
    time.sleep(0.6)
    df_adv = leaguedashteamstats.LeagueDashTeamStats(
        season=SEASON,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]

    return df_off, df_adv


def fetch_clutch_stats():
    """
    Clutch istatistikleri — son 5 dakika, 5 puan fark.
    Kriter 4: Decision-Making Under Pressure için kritik.
    """
    print("Fetching clutch stats...")
    time.sleep(0.6)

    df = leaguedashteamclutch.LeagueDashTeamClutch(
        season=SEASON,
        measure_type_detailed_defense="Base",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]

    return df


def fetch_bench_stats():
    """
    Bench istatistikleri — starter vs bench split.
    Kriter 3: Weak Link Exposure için.
    """
    print("Fetching bench (non-starter) player stats...")
    time.sleep(0.6)

    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
    ).get_data_frames()[0]

    return df


# ============================================================
# KRİTER 1: SHOT CREATION UNDER CONSTRAINT
# ============================================================

# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def _normalize(series, low_val, high_val):
    """
    Pandas Series'i [low_val, high_val] aralığına normalize et.
    low_val > high_val ise ters normalleşme (düşük = iyi).
    """
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([5.0] * len(series), index=series.index)

    if low_val < high_val:
        # Normal: yüksek değer iyi
        return low_val + (series - mn) / (mx - mn) * (high_val - low_val)
    else:
        # Ters: düşük değer iyi
        return high_val + (mx - series) / (mx - mn) * (low_val - high_val)


# Soru: Birinci seçenek kapatıldığında ofans hayatta kalabiliyor mu?
#
# Metrikler:
#   - AST/TOV oranı: Organizasyon kalitesi, baskı altında düşer
#   - TS% (True Shooting): Şut kalitesi, gerçek verimlilik
#   - PACE: Hızlı oyun playoff'ta sürdürülebilir mi?
#   - 3PA/FGA oranı: 3'e bağımlılık → playoff'ta riskli
#
# Playoff insight: Regular season'da pace-driven sistemler
# playoff'ta yavaşlayan tempoya adapte olamaz.

def score_shot_creation(df_off, df_adv):
    """
    Shot Creation Under Constraint skoru.
    Her metriği 0-10 arasına normalize et, ağırlıklı topla.
    """
    results = {}

    for team_name in TEAM_IDS:
        row_off = df_off[df_off["TEAM_NAME"] == team_name]
        row_adv = df_adv[df_adv["TEAM_NAME"] == team_name]

        if row_off.empty or row_adv.empty:
            continue

        row_off = row_off.iloc[0]
        row_adv = row_adv.iloc[0]

        # AST/TOV — yüksek = iyi (top hareketi, düşük risk)
        ast_tov = row_off["AST"] / max(row_off["TOV"], 0.1)

        # TS% = PTS / (2 * (FGA + 0.44 * FTA))
        ts_pct = row_off["PTS"] / (2 * (row_off["FGA"] + 0.44 * row_off["FTA"] + 0.001))

        # PACE — playoff'ta düşer, çok yüksek pace riskli
        pace = row_adv.get("PACE", 100.0)
        # Pace için ters skorlama: 95-100 arası ideal playoff pace
        pace_score = max(0, 10 - abs(pace - 97.5) * 0.5)

        # 3PA bağımlılığı — playoff'ta 3% düşer, yüksek bağımlılık risk
        fg3a_ratio = row_off["FG3A"] / max(row_off["FGA"], 1)
        three_dependency_penalty = max(0, (fg3a_ratio - 0.38) * 20)  # %38 üstü penaltı

        results[team_name] = {
            "ast_tov":    ast_tov,
            "ts_pct":     ts_pct,
            "pace":       pace,
            "pace_score": pace_score,
            "fg3a_ratio": fg3a_ratio,
            "three_dep_penalty": three_dependency_penalty,
        }

    df = pd.DataFrame(results).T

    # Normalize et (0-10)
    df["ast_tov_norm"]  = _normalize(df["ast_tov"], 0, 10)
    df["ts_pct_norm"]   = _normalize(df["ts_pct"], 0, 10)
    df["pace_score_norm"] = df["pace_score"]

    # Ağırlıklı skor
    df["shot_creation_score"] = (
        df["ast_tov_norm"]    * 0.35 +
        df["ts_pct_norm"]     * 0.40 +
        df["pace_score_norm"] * 0.25 -
        df["three_dep_penalty"]
    ).clip(0, 10)

    return df[["ast_tov", "ts_pct", "pace", "fg3a_ratio", "shot_creation_score"]]


# ============================================================
# KRİTER 2: DEFENSIVE SCALABILITY
# ============================================================
# Soru: Defans birden fazla takıma karşı adapte olabiliyor mu?
#
# Playoff insight:
#   "Scheme > personnel regular season"
#   "Personnel > scheme playoffs"
#   Playoff'ta rakip senin sistemini çözer → personnel kalitesi belirler.
#
# Metrikler:
#   - DRTG (Defensive Rating): Her 100 topossession'da izin verilen puan
#   - OPP FG%: Rakip şut yüzdesi
#   - STL + BLK per game: Aktif defans kapasitesi
#   - OPP PTS in paint: İçeriden savunma

def score_defensive_scalability(df_off, df_adv):
    """Defensive Scalability skoru."""
    results = {}

    for team_name in TEAM_IDS:
        row_off = df_off[df_off["TEAM_NAME"] == team_name]
        row_adv = df_adv[df_adv["TEAM_NAME"] == team_name]

        if row_off.empty or row_adv.empty:
            continue

        row_off = row_off.iloc[0]
        row_adv = row_adv.iloc[0]

        drtg  = row_adv.get("DEF_RATING", 112.0)
        stl   = row_off["STL"]
        blk   = row_off["BLK"]
        opp_fg_pct = row_adv.get("OPP_EFG_PCT", 0.52)

        results[team_name] = {
            "drtg":       drtg,
            "stl":        stl,
            "blk":        blk,
            "opp_efg":    opp_fg_pct,
        }

    df = pd.DataFrame(results).T

    # DRTG — düşük = iyi, ters normalize
    df["drtg_norm"]    = _normalize(df["drtg"], 10, 0)      # ters
    df["stl_norm"]     = _normalize(df["stl"], 0, 10)
    df["blk_norm"]     = _normalize(df["blk"], 0, 10)
    df["opp_efg_norm"] = _normalize(df["opp_efg"], 10, 0)   # ters

    df["defensive_score"] = (
        df["drtg_norm"]    * 0.40 +
        df["opp_efg_norm"] * 0.30 +
        df["stl_norm"]     * 0.15 +
        df["blk_norm"]     * 0.15
    )

    return df[["drtg", "stl", "blk", "opp_efg", "defensive_score"]]


# ============================================================
# KRİTER 3: WEAK LINK EXPOSURE
# ============================================================
# Soru: Bench oyuncusu 40 dakika hedef alındığında dayanır mı?
#
# Playoff insight: Rakip analistler en zayıf halkayı tespit eder
# ve play call'ları o oyuncuya karşı organize eder.
#
# Metrikler:
#   - Bench scoring: İkinci ünite ofansif üretimi
#   - Starter/bench PTS split: Bağımlılık konsantrasyonu
#   - Bench MIN%: Bench'e güvenme oranı
#   - Roster depth score: Değiştirilebilir starter var mı?

def score_weak_link(df_player):
    """
    Weak Link Exposure skoru.
    Bench kalitesini ve kadro derinliğini ölçer.
    """
    results = {}

    
    for team_name, team_id in TEAM_IDS.items():
            team_players = df_player[df_player["TEAM_ID"] == team_id].copy()

            if team_players.empty:
                continue

            # Sabit threshold yerine takım içi sıralama
            # Her takımın en çok oynayan 8 oyuncusu = rotation
            # İlk 5 = starter, sonraki 3 = core bench
            team_players = team_players[team_players["MIN"] >= 6].copy()
            team_players = team_players.sort_values("MIN", ascending=False)

            starters = team_players.head(5)
            bench    = team_players.iloc[5:9]   # 6-9. en çok oynayan

            starter_pts = starters["PTS"].sum()
            bench_pts   = bench["PTS"].sum() if len(bench) > 0 else 0
            total_pts   = starter_pts + bench_pts + 0.001

            bench_contribution = bench_pts / total_pts
            bench_pm = bench["PLUS_MINUS"].mean() if len(bench) > 0 and "PLUS_MINUS" in bench.columns else 0
            depth_count = len(team_players)

            results[team_name] = {
                "bench_pts":          bench_pts,
                "bench_contribution": bench_contribution,
                "bench_pm":           bench_pm,
                "depth_count":        depth_count,
            }

    df = pd.DataFrame(results).T

    df["bench_contrib_norm"] = _normalize(df["bench_contribution"], 0, 10)
    df["bench_pm_norm"]      = _normalize(df["bench_pm"], 0, 10)
    df["depth_norm"]         = _normalize(df["depth_count"], 0, 10)

    df["depth_score"] = (
        df["bench_contrib_norm"] * 0.35 +
        df["bench_pm_norm"]      * 0.40 +
        df["depth_norm"]         * 0.25
    )

    return df[["bench_pts", "bench_contribution", "bench_pm", "depth_count", "depth_score"]]


# ============================================================
# KRİTER 4: DECISION-MAKING UNDER PRESSURE
# ============================================================
# Soru: Yıldız oyuncular entropy azaltıyor mu, artırıyor mu?
#
# Playoff insight: En iyi playoff oyuncuları baskı altında
# turnover yapmaz, doğru kararı verir, skoru kapatır.
#
# Metrikler:
#   - Clutch W% (son 5 dk, 5 puan fark)
#   - Clutch +/-
#   - Clutch TOV rate
#   - Clutch TS%

def score_clutch_performance(df_clutch):
    """Decision-Making Under Pressure skoru."""
    results = {}

    for team_name in TEAM_IDS:
        row = df_clutch[df_clutch["TEAM_NAME"] == team_name]

        if row.empty:
            continue

        row = row.iloc[0]

        clutch_gp    = max(row.get("GP", 1), 1)
        clutch_wins  = row.get("W", 0)
        clutch_wPct  = clutch_wins / clutch_gp
        clutch_pm    = row.get("PLUS_MINUS", 0)
        clutch_tov   = row.get("TOV", 2.0)
        clutch_pts   = row.get("PTS", 10.0)

        results[team_name] = {
            "clutch_wpct": clutch_wPct,
            "clutch_pm":   clutch_pm,
            "clutch_tov":  clutch_tov,
            "clutch_pts":  clutch_pts,
        }

    df = pd.DataFrame(results).T

    df["cwpct_norm"] = _normalize(df["clutch_wpct"], 0, 10)
    df["cpm_norm"]   = _normalize(df["clutch_pm"], 0, 10)
    df["ctov_norm"]  = _normalize(df["clutch_tov"], 10, 0)  # ters

    df["clutch_score"] = (
        df["cwpct_norm"] * 0.50 +
        df["cpm_norm"]   * 0.30 +
        df["ctov_norm"]  * 0.20
    )

    return df[["clutch_wpct", "clutch_pm", "clutch_tov", "clutch_score"]]


# ============================================================
# KRİTER 5: ROSTER OPTIONALITY
# ============================================================
# Soru: Taktik değiştirebiliyor mu? Farklı lineup'lar işe yarıyor mu?
#
# Playoff insight: Seride rakip seni çözdükten sonra
# "Plan B" var mı? Farklı boyut, tempo, stil oynayabiliyor musun?
#
# Metrikler:
#   - Net Rating (genel):  güçlü takım = seçenek çok
#   - ORtg vs DRtg dengesi: Two-way team mi, tek taraflı mı?
#   - W/L yakın maçlarda:  Farklı senaryolarda kazanabiliyor mu?

def score_roster_optionality(df_off, df_adv):
    """Roster Optionality skoru."""
    results = {}

    for team_name in TEAM_IDS:
        row_off = df_off[df_off["TEAM_NAME"] == team_name]
        row_adv = df_adv[df_adv["TEAM_NAME"] == team_name]

        if row_off.empty or row_adv.empty:
            continue

        row_off = row_off.iloc[0]
        row_adv = row_adv.iloc[0]

        net_rtg  = row_adv.get("NET_RATING", 0.0)
        off_rtg  = row_adv.get("OFF_RATING", 112.0)
        def_rtg  = row_adv.get("DEF_RATING", 112.0)
        wins     = row_off["W"]
        losses   = row_off["L"]
        win_pct  = wins / max(wins + losses, 1)

        # Two-way balance: Ofans ve defans dengesi
        # En iyi playoff takımları her ikisinde de iyi
        ortg_norm_val = (off_rtg - 105) / 15    # ~0-1 arası
        drtg_norm_val = (120 - def_rtg) / 15    # ters, ~0-1 arası
        two_way_score = (ortg_norm_val + drtg_norm_val) / 2

        results[team_name] = {
            "net_rtg":       net_rtg,
            "win_pct":       win_pct,
            "two_way_score": two_way_score,
            "off_rtg":       off_rtg,
            "def_rtg":       def_rtg,
        }

    df = pd.DataFrame(results).T

    df["net_rtg_norm"]    = _normalize(df["net_rtg"], 0, 10)
    df["win_pct_norm"]    = _normalize(df["win_pct"], 0, 10)
    df["two_way_norm"]    = _normalize(df["two_way_score"], 0, 10)

    df["optionality_score"] = (
        df["net_rtg_norm"]  * 0.40 +
        df["two_way_norm"]  * 0.35 +
        df["win_pct_norm"]  * 0.25
    )

    return df[["net_rtg", "win_pct", "two_way_score", "off_rtg", "def_rtg", "optionality_score"]]


# ============================================================
# GENEL PLAYOFF VİABİLİTY SKORU
# ============================================================
# Kriterlerin ağırlıkları — playoff'ta hangisi daha belirleyici?
#
# Defensive Scalability en ağır: Playoff şampiyonları defansla kazanır
# Clutch en kritik ikinci: Seri karar anlarında fark yaratır
# Shot Creation: Zor şut üretimi seriyi uzatır
# Weak Link: Hedef alınma riski ciddi kayıplara neden olabilir
# Roster Optionality: Uzun vadeli seri adaptasyonu

CRITERION_WEIGHTS = {
    "shot_creation_score": 0.20,
    "defensive_score":     0.30,
    "depth_score":         0.15,
    "clutch_score":        0.25,
    "optionality_score":   0.10,
}

def calculate_viability_score(shot_df, def_df, depth_df, clutch_df, opt_df):
    """
    Tüm kriterleri birleştir, genel playoff viability skoru hesapla.
    """
    all_teams = list(TEAM_IDS.keys())

    scores = pd.DataFrame(index=all_teams)

    for col, df in [
        ("shot_creation_score", shot_df),
        ("defensive_score",     def_df),
        ("depth_score",         depth_df),
        ("clutch_score",        clutch_df),
        ("optionality_score",   opt_df),
    ]:
        scores[col] = df[col]

    # Ağırlıklı genel skor (0-10)
    scores["playoff_viability"] = sum(
        scores[col] * weight
        for col, weight in CRITERION_WEIGHTS.items()
    )

    # Seeding bonusu: Ev sahibi avantajı gerçek bir etken
    for conf, teams in [("East", PLAYOFF_TEAMS_EAST), ("West", PLAYOFF_TEAMS_WEST)]:
        for seed, team in teams.items():
            if team in scores.index:
                # 1-2. tohum: +0.3, 3-4: +0.1, 5-8: 0
                bonus = 0.3 if seed <= 2 else (0.1 if seed <= 4 else 0)
                scores.loc[team, "playoff_viability"] += bonus

    scores["playoff_viability"] = scores["playoff_viability"].clip(0, 10)
    
    # Konferans bilgisi ekle
    conf_map = {t: "East" for t in PLAYOFF_TEAMS_EAST.values()}
    conf_map.update({t: "West" for t in PLAYOFF_TEAMS_WEST.values()})
    scores["conference"] = scores.index.map(lambda x: conf_map.get(x, "Unknown"))

    seed_map = {t: s for s, t in PLAYOFF_TEAMS_EAST.items()}
    seed_map.update({t: s for s, t in PLAYOFF_TEAMS_WEST.items()})
    scores["seed"] = scores.index.map(lambda x: seed_map.get(x, 9))

    return scores.sort_values("playoff_viability", ascending=False)


# ============================================================
# GÖRSELLEŞTİRME
# ============================================================

def visualize_results(scores, shot_df, def_df, clutch_df):
    """Playoff viability dashboard."""
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0d1117")

    # Renk paleti
    east_color = "#4fc3f7"   # mavi — East
    west_color = "#f97316"   # turuncu — West
    champion_color = "#ffd700"  # altın

    # ── Panel 1: Genel Playoff Viability Sıralaması ─────────
    ax1 = fig.add_subplot(2, 2, (1, 2))
    ax1.set_facecolor("#161b22")

    sorted_scores = scores.sort_values("playoff_viability", ascending=True)
    colors = [east_color if c == "East" else west_color
              for c in sorted_scores["conference"]]

    bars = ax1.barh(range(len(sorted_scores)),
                   sorted_scores["playoff_viability"],
                   color=colors, alpha=0.85, height=0.7)

    ax1.set_yticks(range(len(sorted_scores)))
    ax1.set_yticklabels(
        [f"{'E' if c=='East' else 'W'}{s} — {t}"
         for t, s, c in zip(sorted_scores.index,
                            sorted_scores["seed"],
                            sorted_scores["conference"])],
        fontsize=9, color="white"
    )

    for i, (bar, (team, row)) in enumerate(zip(bars, sorted_scores.iterrows())):
        ax1.text(bar.get_width() + 0.05, i,
                f"{row['playoff_viability']:.2f}",
                va="center", fontsize=8, color="white")

    ax1.set_xlabel("Playoff Viability Score (0-10)", color="white")
    ax1.set_title("2025-26 NBA Playoff Viability Model\n"
                 "Mavi = East | Turuncu = West",
                 color="white", fontsize=13, fontweight="bold")
    ax1.tick_params(colors="white")
    ax1.spines["bottom"].set_color("#30363d")
    ax1.spines["left"].set_color("#30363d")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xlim(0, 11)

    # ── Panel 2: Kriter Radar (Top 6) ───────────────────────
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.set_facecolor("#161b22")

    criteria = ["Shot\nCreation", "Defense", "Depth", "Clutch", "Optionality"]
    score_cols = ["shot_creation_score", "defensive_score",
                  "depth_score", "clutch_score", "optionality_score"]

    top6 = scores.nlargest(6, "playoff_viability")

    x = np.arange(len(criteria))
    width = 0.13
    cmap = plt.cm.get_cmap("tab10")

    for i, (team, row) in enumerate(top6.iterrows()):
        vals = [row[c] for c in score_cols]
        color = east_color if row["conference"] == "East" else west_color
        ax2.bar(x + i * width, vals, width,
               label=team.split()[-1],
               color=cmap(i), alpha=0.8)

    ax2.set_xticks(x + width * 2.5)
    ax2.set_xticklabels(criteria, color="white", fontsize=8)
    ax2.set_ylabel("Skor (0-10)", color="white")
    ax2.set_title("Top 6 — Kriter Karşılaştırması",
                 color="white", fontsize=11)
    ax2.legend(fontsize=7, loc="upper right",
               facecolor="#161b22", labelcolor="white")
    ax2.tick_params(colors="white")
    ax2.spines["bottom"].set_color("#30363d")
    ax2.spines["left"].set_color("#30363d")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ── Panel 3: Shot Creation vs Defense scatter ────────────
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.set_facecolor("#161b22")

    for team, row in scores.iterrows():
        c = east_color if row["conference"] == "East" else west_color
        ax3.scatter(row["shot_creation_score"],
                   row["defensive_score"],
                   s=120, color=c, alpha=0.8, zorder=3)
        ax3.annotate(team.split()[-1],
                    (row["shot_creation_score"], row["defensive_score"]),
                    fontsize=7, color="white", alpha=0.9,
                    xytext=(4, 4), textcoords="offset points")

    # Ortalama çizgileri
    avg_shot = scores["shot_creation_score"].mean()
    avg_def  = scores["defensive_score"].mean()
    ax3.axvline(avg_shot, color="#555", linestyle="--", alpha=0.5)
    ax3.axhline(avg_def, color="#555", linestyle="--", alpha=0.5)

    ax3.text(avg_shot + 0.1, scores["defensive_score"].max() * 0.98,
            "Avg", color="#888", fontsize=7)

    ax3.set_xlabel("Shot Creation Score", color="white")
    ax3.set_ylabel("Defensive Score", color="white")
    ax3.set_title("Ofans vs Defans Dengesi\n(Sağ üst = Two-Way elite)",
                 color="white", fontsize=11)
    ax3.tick_params(colors="white")
    ax3.spines["bottom"].set_color("#30363d")
    ax3.spines["left"].set_color("#30363d")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    plt.tight_layout(pad=2.5)
    plt.savefig("playoff_viability_2026.png", dpi=150,
               bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print("Grafik kaydedildi: playoff_viability_2026.png")


# ============================================================
# BRACKET TAHMİNİ
# ============================================================

def predict_bracket(scores):
    """
    Playoff viability skorlarına göre bracket tahmini.
    Her seride üst seeded takım ile karşılaştır,
    viability farkı > 0.5 ise sürpriz olası değil.
    """
    print("\n" + "=" * 60)
    print("BRACKET TAHMİNİ — 2025-26 NBA PLAYOFFS")
    print("=" * 60)

    for conf, teams in [("EAST", PLAYOFF_TEAMS_EAST), ("WEST", PLAYOFF_TEAMS_WEST)]:
        print(f"\n── {conf} ──────────────────────────────")

        # İlk tur maçupları: 1v8, 2v7, 3v6, 4v5
        matchups = [(1, 8), (2, 7), (3, 6), (4, 5)]

        conf_winners = {}

        for high_seed, low_seed in matchups:
            team_high = teams.get(high_seed)
            team_low  = teams.get(low_seed)

            if not team_high or not team_low:
                continue

            score_high = scores.loc[team_high, "playoff_viability"] if team_high in scores.index else 5.0
            score_low  = scores.loc[team_low,  "playoff_viability"] if team_low  in scores.index else 5.0

            diff = score_high - score_low
            winner = team_high if score_high >= score_low else team_low

            # Seri tahmini
            if abs(diff) >= 1.5:
                series = "4-1"
            elif abs(diff) >= 0.7:
                series = "4-2"
            else:
                series = "4-3 (yakın seri)"

            print(f"  ({high_seed}) {team_high:<28} [{score_high:.2f}]")
            print(f"  ({low_seed}) {team_low:<28} [{score_low:.2f}]")
            print(f"  → Tahmin: {winner} wins {series}")
            if abs(diff) < 0.5:
                print(f"  ⚠ UPSET RİSKİ — fark çok küçük ({diff:.2f})")
            print()

            conf_winners[high_seed] = (winner, max(score_high, score_low))

        # Konferans finalisti tahmini
        if len(conf_winners) >= 4:
            finalist_1_team, finalist_1_score = max(
                [conf_winners.get(1, ("?", 0)), conf_winners.get(4, ("?", 0))],
                key=lambda x: x[1]
            )
            finalist_2_team, finalist_2_score = max(
                [conf_winners.get(2, ("?", 0)), conf_winners.get(3, ("?", 0))],
                key=lambda x: x[1]
            )
            conf_finalist = finalist_1_team if finalist_1_score >= finalist_2_score else finalist_2_team
            print(f"  🏆 {conf} Finals tahmini: {conf_finalist}")

    # Şampiyonluk tahmini
    print("\n── NBA FİNALLERİ TAHMİNİ ─────────────────")
    top2 = scores.nlargest(2, "playoff_viability")
    champion = top2.index[0]
    finalist = top2.index[1]
    print(f"  {champion} vs {finalist}")
    print(f"  🏆 Şampiyon tahmini: {champion}")
    print(f"     Playoff Viability: {scores.loc[champion, 'playoff_viability']:.2f}/10")


# ============================================================
# MAIN
# ============================================================

def main():
    print("NBA Playoff Viability Model — 2025-26")
    print("=" * 50)

    # Veri çek
    df_off, df_adv = fetch_base_team_stats()
    df_clutch       = fetch_clutch_stats()
    df_player       = fetch_bench_stats()

    print(f"Veri yüklendi: {len(df_off)} takım")

    # Kriter skorlarını hesapla
    print("\nKriter skorları hesaplanıyor...")
    shot_df   = score_shot_creation(df_off, df_adv)
    def_df    = score_defensive_scalability(df_off, df_adv)
    depth_df  = score_weak_link(df_player)
    clutch_df = score_clutch_performance(df_clutch)
    opt_df    = score_roster_optionality(df_off, df_adv)

    # Genel skor
    scores = calculate_viability_score(shot_df, def_df, depth_df, clutch_df, opt_df)

    # Detaylı çıktı
    print("\n" + "=" * 60)
    print("PLAYOFF VİABİLİTY SKORLARI — TÜM KRİTERLER")
    print("=" * 60)

    detailed = pd.concat([
        scores[["conference", "seed", "playoff_viability"]],
        shot_df[["shot_creation_score"]],
        def_df[["defensive_score"]],
        depth_df[["depth_score"]],
        clutch_df[["clutch_score"]],
        opt_df[["optionality_score"]],
    ], axis=1).dropna()

    detailed = detailed.sort_values("playoff_viability", ascending=False)
    print(detailed.round(2).to_string())

    # Bracket tahmini
    predict_bracket(scores)

    # Görselleştir
    visualize_results(scores, shot_df, def_df, clutch_df)

    # CSV kaydet
    detailed.to_csv("playoff_viability_2026.csv")
    print("\nVeri kaydedildi: playoff_viability_2026.csv")

    print("\n" + "=" * 50)
    print("Git push için:")
    print("  git add playoff_model.py playoff_viability_2026.png")
    print("  git commit -m 'Add playoff viability model 2026'")
    print("  git push")


if __name__ == "__main__":
    main()
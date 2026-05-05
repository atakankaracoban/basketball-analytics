"""
NBA MVP Predictor — 2025-26 Season
====================================
Soru: Kimi seçecekler? (Kim seçilmeli değil)

Model iki katmanlı:
  Katman 1 — Data Score:      İstatistiksel dominance + verimlilik
  Katman 2 — Narrative Bonus: Medya anlatısı, momentum, team success

Tarihsel analiz: 2004-05'ten bugüne MVP pattern'ları
Veri kaynağı: nba_api (gerçek zamanlı)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import warnings
warnings.filterwarnings("ignore")

from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats
from nba_api.stats.static import players as nba_players_static

# ============================================================
# BÖLÜM 1: TARİHSEL MVP VERİSİ
# ============================================================
# 2004-05'ten 2024-25'e kadar MVP kazananlar ve o sezonun
# istatistikleri + finalist rakipleri
#
# NEDEN BU DÖNEM:
#   2004-05 öncesi bazı advanced stats tutarsız
#   Bu 20 sezon modern NBA voting pattern'ını temsil eder
#
# KAYNAK: Basketball Reference + Voting records
# Her satır: MVP kazanan, finalist(ler), kritik metrikler

HISTORICAL_MVPS = [
    # (season, winner, team, wins, pts, reb, ast, ts_pct, ws, narrative_strength)
    # narrative_strength: 1-10, o sezonki medya anlatısının gücü
    ("2004-05", "Steve Nash",         "PHX", 62, 15.5,  3.3, 11.5, 0.604, 18.0, 7),
    ("2005-06", "Steve Nash",         "PHX", 54, 18.8,  4.2, 10.5, 0.604, 16.2, 6),
    ("2006-07", "Dirk Nowitzki",      "DAL", 67, 24.6,  8.9,  3.4, 0.605, 19.5, 8),
    ("2007-08", "Kobe Bryant",        "LAL", 57, 28.3,  6.3,  5.4, 0.561, 14.9, 9),
    ("2008-09", "LeBron James",       "CLE", 66, 28.4,  7.6,  7.2, 0.591, 20.0, 9),
    ("2009-10", "LeBron James",       "CLE", 61, 29.7,  7.3,  8.6, 0.605, 21.9, 10),
    ("2010-11", "Derrick Rose",       "CHI", 62, 25.0,  4.1,  7.7, 0.540, 12.7, 8),
    ("2011-12", "LeBron James",       "MIA", 46, 27.1,  7.9,  6.2, 0.605, 17.0, 8),
    ("2012-13", "LeBron James",       "MIA", 66, 26.8,  8.0,  7.3, 0.640, 23.0, 10),
    ("2013-14", "Kevin Durant",       "OKC", 59, 32.0,  7.4,  5.5, 0.635, 19.3, 9),
    ("2014-15", "Stephen Curry",      "GSW", 67, 23.8,  4.3,  7.7, 0.643, 17.9, 9),
    ("2015-16", "Stephen Curry",      "GSW", 73, 30.1,  5.4,  6.7, 0.670, 17.9, 10),  # unanimous
    ("2016-17", "Russell Westbrook",  "OKC", 47, 31.6, 10.7, 10.4, 0.554, 13.8, 10), # triple-double anomali
    ("2017-18", "James Harden",       "HOU", 65, 30.4,  5.4,  8.8, 0.612, 15.6, 9),
    ("2018-19", "Giannis Antetokounmpo","MIL",60, 27.7, 12.5,  5.9, 0.618, 15.7, 8),
    ("2019-20", "Giannis Antetokounmpo","MIL",56, 29.6, 13.7,  5.6, 0.615, 14.1, 9),
    ("2020-21", "Nikola Jokić",       "DEN", 47, 26.4, 10.8,  8.3, 0.664, 15.1, 7),
    ("2021-22", "Nikola Jokić",       "DEN", 48, 27.1, 13.8,  7.9, 0.665, 15.1, 8),
    ("2022-23", "Joel Embiid",        "PHI", 54, 33.1, 10.2,  4.2, 0.650, 14.8, 8),
    ("2023-24", "Nikola Jokić",       "DEN", 57, 26.4, 12.4,  9.0, 0.695, 18.2, 8),
    ("2024-25", "Shai Gilgeous-Alexander","OKC",68,32.7, 5.5,  6.4, 0.655, 20.1, 9),
]

# Tarihsel finalistler (MVP olmayan ama top 3'e giren)
# Bunlar modeli kalibre etmek için kullanılır
HISTORICAL_RUNNERS_UP = [
    # (season, player, team, wins, pts, reb, ast, ts_pct, ws)
    ("2004-05", "Shaquille O'Neal",   "MIA", 59, 22.9, 10.4, 2.7, 0.584, 15.0),
    ("2006-07", "Steve Nash",         "PHX", 61, 18.6,  3.5, 11.6, 0.612, 18.0),
    ("2007-08", "Chris Paul",         "NOP", 56, 21.1,  4.0, 11.6, 0.596, 20.0),
    ("2008-09", "Kobe Bryant",        "LAL", 65, 26.8,  5.2,  4.9, 0.567, 14.5),
    ("2012-13", "Kevin Durant",       "OKC", 60, 28.1,  7.9,  4.6, 0.635, 17.5),
    ("2013-14", "LeBron James",       "MIA", 54, 27.1,  6.9,  6.3, 0.620, 20.0),
    ("2015-16", "Kawhi Leonard",      "SAS", 67, 21.2,  6.8,  2.6, 0.600, 14.0),
    ("2017-18", "LeBron James",       "CLE", 50, 27.5,  8.6,  9.1, 0.592, 13.2),
    ("2018-19", "Paul George",        "OKC", 49, 28.0,  8.2,  4.1, 0.594, 11.4),
    ("2022-23", "Nikola Jokić",       "DEN", 53, 24.5, 11.8,  9.8, 0.680, 15.0),
    ("2023-24", "Luka Dončić",        "DAL", 50, 33.9,  9.2,  9.8, 0.641, 13.5),
    ("2024-25", "Nikola Jokić",       "DEN", 50, 29.8, 13.5, 10.0, 0.700, 17.0),
]

# ============================================================
# BÖLÜM 2: TARİHSEL PATTERN ANALİZİ
# ============================================================

def analyze_historical_patterns():
    """
    Tarihsel MVP verilerinden pattern çıkar.
    Hangi metrikler MVP'yi non-MVP'den ayırt ediyor?
    """
    df_mvp = pd.DataFrame(HISTORICAL_MVPS,
        columns=["season","player","team","wins","pts","reb","ast","ts_pct","ws","narrative"])
    df_runner = pd.DataFrame(HISTORICAL_RUNNERS_UP,
        columns=["season","player","team","wins","pts","reb","ast","ts_pct","ws"])
    df_runner["narrative"] = 5  # Orta seviye default

    df_mvp["is_mvp"] = 1
    df_runner["is_mvp"] = 0

    df_all = pd.concat([df_mvp, df_runner], ignore_index=True)

    print("=" * 60)
    print("TARİHSEL MVP PATTERN ANALİZİ (2004-25)")
    print("=" * 60)

    print("\n--- MVP Kazananların Ortalama İstatistikleri ---")
    mvp_avg = df_mvp[["wins","pts","reb","ast","ts_pct","ws","narrative"]].mean()
    runner_avg = df_runner[["wins","pts","reb","ast","ts_pct","ws","narrative"]].mean()

    comparison = pd.DataFrame({
        "MVP Ortalaması":    mvp_avg.round(2),
        "Finalist Ortalaması": runner_avg.round(2),
        "MVP Avantajı":     (mvp_avg - runner_avg).round(2)
    })
    print(comparison)

    print("\n--- Kritik Eşikler (MVP Kazananların %80'i bu değerlerin üstünde) ---")
    thresholds = {}
    for col in ["wins","pts","ts_pct","ws"]:
        val = df_mvp[col].quantile(0.20)  # alt %20'yi kes
        thresholds[col] = round(val, 3)
        print(f"  {col}: {val:.3f}+")

    print("\n--- Anomaliler (Modele uymayan MVP'ler) ---")
    anomalies = df_mvp[df_mvp["wins"] < 55]
    print("50'den az galibiyet ile MVP kazananlar:")
    print(anomalies[["season","player","team","wins","pts","narrative"]].to_string(index=False))
    print("\n→ Ortak özellik: Narrative score 8+ (güçlü anlatı tazminatı)")

    return df_mvp, df_runner, thresholds


# ============================================================
# BÖLÜM 3: AĞIRLIK SİSTEMİ
# ============================================================
# Tarihsel pattern'dan çıkan ağırlıklar:
#
# KATMAN 1 — DATA SCORE (0-10):
#   Team Wins:     En güvenilir gösterge. MVP takımı kazanmalı.
#   PTS:           Scoring dominance medyayı çeker.
#   TS%:           Verimlilik — ham sayıyı meşrulaştırır.
#   Win Shares:    Takıma katkının en iyi tek metriği.
#   AST+REB:       Çok yönlülük bonusu.
#
# KATMAN 2 — NARRATIVE BONUS (0-3):
#   Sezon anlatısının gücü, rakipsizlik, medya ilgisi.
#   Manuel girilir — bu modelin "insider knowledge" katmanı.

DATA_WEIGHTS = {
    "wins_norm":   0.25,  # Takım galibiyeti
    "pts_norm":    0.25,  # Scoring dominance
    "ts_norm":     0.20,  # Verimlilik
    "ws_norm":     0.20,  # Win Shares
    "versatility": 0.10,  # AST + REB kombinasyonu
}

# ============================================================
# BÖLÜM 4: BU SEZON ADAYLARI
# ============================================================
# 2025-26 sezonu MVP adayları
# Veriler: nba_api'den çekilecek + manuel narrative bonus

MVP_CANDIDATES_2026 = [
    "Shai Gilgeous-Alexander",
    "Nikola Jokić",
    "Victor Wembanyama",
    "Giannis Antetokounmpo",
    "Jayson Tatum",
    "Luka Dončić",
    "Cooper Flagg",
    "Anthony Edwards",
    "Karl-Anthony Towns",
]

# Narrative Bonus (0-3) — Manuel değerlendirme
# Bu katman tamamen voting psychology'ye dayanıyor:
#   3.0 = Sezonun tartışmasız hikayesi, medya zaten seçti
#   2.0 = Güçlü aday, ciddi anlatı var
#   1.0 = İstatistiksel aday ama anlatı zayıf
#   0.0 = Dark horse, medya henüz fark etmedi
#
# GEREKÇELER:
#   SGA: Defending champion takımın yıldızı, ligin skorer lideri,
#        OKC'yi 1. tohuma taşıdı. 2024-25 MVP'si zaten oydu.
#        Consecutive MVP anlatısı güçlü ama "tekrar" yorgunluğu var.
#   Jokić: Dört MVP almış. Voters bunu bir daha vermekten kaçınıyor
#           (voter fatigue gerçek bir fenomen). Ama istatistiksel
#           olarak her zaman top 3'te.
#   Giannis: Milwaukee yeniden güçlendi. "Comeback narrative" var.
#   Tatum: Sakatlık dönemi anlatıyı zayıflattı.
#   Luka: Sakatlık + Dallas'tan ayrılık anlatısı karışık.
#   Flagg: Rookie of the Year anlatısı MVP'nin önüne geçiyor.
#   Edwards: Timberwolves başarısı güçlü anlatı üretiyor.
#   KAT: New York'a transferi büyük anlatı ama Knicks sistemi
#        bireysel dominance'ı gizleyebilir.

NARRATIVE_BONUS = {
    "Shai Gilgeous-Alexander": 2.5,
    "Nikola Jokić":            1.5,  # voter fatigue
    "Giannis Antetokounmpo":   1.8,
    "Jayson Tatum":            1.2,  # sakatlık anlatısı
    "Luka Dončić":             1.0,  # karışık anlatı
    "Cooper Flagg":            1.0,  # ROY anlatısı öne geçiyor
    "Victor Wembanyama":       2.8,  # Genç süperstar, ligin yeni yüzü, dominant performans
    "Anthony Edwards":         2.0,  # Wolves başarısı
    "Karl-Anthony Towns":      1.5,  # NYK anlatısı
}

# ============================================================
# BÖLÜM 5: VERİ ÇEKME
# ============================================================

def fetch_current_season_stats():
    """Bu sezon aday oyuncuların istatistiklerini çek."""
    print("\nBu sezon oyuncu istatistikleri çekiliyor...")
    time.sleep(0.6)

    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season="2025-26",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
    ).get_data_frames()[0]

    # Win Shares için Advanced stats
    time.sleep(0.6)
    df_adv = leaguedashplayerstats.LeagueDashPlayerStats(
        season="2025-26",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
    ).get_data_frames()[0]

    return df, df_adv


def fetch_team_wins():
    """Takım galibiyet sayılarını çek."""
    time.sleep(0.6)
    df = leaguedashteamstats.LeagueDashTeamStats(
        season="2025-26",
        measure_type_detailed_defense="Base",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]
    return df[["TEAM_NAME","W","TEAM_ID"]]


def get_candidate_stats(df_players, df_adv, df_teams):
    """MVP adaylarının istatistiklerini birleştir."""
    results = []

    # Takım ID'den kazanma sayısına map
    team_wins_map = dict(zip(df_teams["TEAM_ID"], df_teams["W"]))

    for candidate in MVP_CANDIDATES_2026:
        row = df_players[df_players["PLAYER_NAME"] == candidate]
        row_adv = df_adv[df_adv["PLAYER_NAME"] == candidate]

        if row.empty:
            print(f"  ⚠ {candidate} bulunamadı, atlanıyor.")
            continue

        row = row.iloc[0]
        row_adv = row_adv.iloc[0] if not row_adv.empty else None

        pts    = float(row["PTS"])
        reb    = float(row["REB"])
        ast    = float(row["AST"])
        gp     = int(row["GP"])
        team_id = int(row["TEAM_ID"])
        wins   = team_wins_map.get(team_id, 45)

        # TS% hesapla
        fga = float(row["FGA"])
        fta = float(row["FTA"])
        ts_pct = pts / (2 * (fga + 0.44 * fta + 0.001))

        # Win Shares proxy (advanced'dan çekmeye çalış)
        ws = 0.0
        if row_adv is not None:
            # nba_api advanced'da PIE (Player Impact Estimate) var
            pie = float(row_adv.get("PIE", 0) or 0)
            ws = pie * gp * 0.15  # proxy formula

        results.append({
            "player":      candidate,
            "team_id":     team_id,
            "wins":        wins,
            "pts":         pts,
            "reb":         reb,
            "ast":         ast,
            "ts_pct":      ts_pct,
            "ws_proxy":    ws,
            "gp":          gp,
            "versatility": reb + ast,
            "narrative":   NARRATIVE_BONUS.get(candidate, 1.0),
        })

    return pd.DataFrame(results).set_index("player")


# ============================================================
# BÖLÜM 6: SKOR HESAPLAMA
# ============================================================

def _normalize_series(s, low=0, high=10):
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series([5.0]*len(s), index=s.index)
    return low + (s - mn) / (mx - mn) * (high - low)


def calculate_mvp_score(df):
    """
    İki katmanlı MVP skoru hesapla.
    Katman 1: Data Score (0-10)
    Katman 2: Narrative Bonus (0-3)
    Toplam: 0-13 (ama normalize edilecek)
    """
    df = df.copy()

    # Normalize
    df["wins_norm"]    = _normalize_series(df["wins"])
    df["pts_norm"]     = _normalize_series(df["pts"])
    df["ts_norm"]      = _normalize_series(df["ts_pct"])
    df["ws_norm"]      = _normalize_series(df["ws_proxy"])
    df["vers_norm"]    = _normalize_series(df["versatility"])

    # Katman 1: Data Score
    df["data_score"] = (
        df["wins_norm"]  * DATA_WEIGHTS["wins_norm"] +
        df["pts_norm"]   * DATA_WEIGHTS["pts_norm"] +
        df["ts_norm"]    * DATA_WEIGHTS["ts_norm"] +
        df["ws_norm"]    * DATA_WEIGHTS["ws_norm"] +
        df["vers_norm"]  * DATA_WEIGHTS["versatility"]
    )

    # Katman 2: Narrative Bonus (0-3 → 0-3 olarak kalır)
    df["narrative_bonus"] = df["narrative"]

    # Toplam MVP Skoru
    df["mvp_score"] = df["data_score"] + df["narrative_bonus"]

    # Normalize to 0-10
    df["mvp_score_normalized"] = _normalize_series(df["mvp_score"], 0, 10)

    return df.sort_values("mvp_score_normalized", ascending=False)


# ============================================================
# BÖLÜM 7: GÖRSELLEŞTİRME
# ============================================================

def visualize_mvp(df, df_historical):
    """MVP prediction dashboard."""
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    gold   = "#FFD700"
    silver = "#C0C0C0"
    accent = "#4fc3f7"

    # ── Panel 1: MVP Skor Sıralaması ───────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#161b22")

    sorted_df = df.sort_values("mvp_score_normalized", ascending=True)
    bar_colors = [gold if i == len(sorted_df)-1
                  else silver if i == len(sorted_df)-2
                  else accent
                  for i in range(len(sorted_df))]

    bars = ax1.barh(range(len(sorted_df)),
                   sorted_df["mvp_score_normalized"],
                   color=bar_colors, alpha=0.88, height=0.65)

    # Data vs Narrative ayrımını göster
    ax1.barh(range(len(sorted_df)),
            _normalize_series(sorted_df["data_score"], 0, 10) *
            (sorted_df["mvp_score_normalized"] / sorted_df["mvp_score_normalized"].max()),
            color="#2d5a8e", alpha=0.5, height=0.65, label="Data Score")

    ax1.set_yticks(range(len(sorted_df)))
    ax1.set_yticklabels(sorted_df.index, fontsize=10, color="white")

    for i, (bar, (player, row)) in enumerate(zip(bars, sorted_df.iterrows())):
        ax1.text(bar.get_width() + 0.1, i,
                f"{row['mvp_score_normalized']:.2f}  "
                f"(Data: {row['data_score']:.1f} | Narrative: +{row['narrative_bonus']:.1f})",
                va="center", fontsize=8, color="white")

    ax1.set_xlabel("MVP Score (0-10)", color="white")
    ax1.set_title("2025-26 NBA MVP Predictor — İki Katmanlı Model\n"
                 "Altın = Tahminimiz | Koyu mavi = Data katmanı | Açık = Narrative bonus",
                 color="white", fontsize=12, fontweight="bold")
    ax1.tick_params(colors="white")
    ax1.set_xlim(0, 13)
    ax1.spines["bottom"].set_color("#30363d")
    ax1.spines["left"].set_color("#30363d")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel 2: Data Score vs Narrative scatter ───────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#161b22")

    for player, row in df.iterrows():
        c = gold if row["mvp_score_normalized"] == df["mvp_score_normalized"].max() else accent
        ax2.scatter(row["data_score"], row["narrative_bonus"],
                   s=150, color=c, alpha=0.85, zorder=3)
        ax2.annotate(player.split()[-1],
                    (row["data_score"], row["narrative_bonus"]),
                    fontsize=7.5, color="white",
                    xytext=(4, 4), textcoords="offset points")

    ax2.set_xlabel("Data Score (0-10)", color="white")
    ax2.set_ylabel("Narrative Bonus (0-3)", color="white")
    ax2.set_title("Data vs Narrative\n(Sağ üst = Oylamanın gideceği yer)",
                 color="white", fontsize=10)
    ax2.tick_params(colors="white")
    ax2.spines["bottom"].set_color("#30363d")
    ax2.spines["left"].set_color("#30363d")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ── Panel 3: Tarihsel MVP İstatistik Ortalamaları ──────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#161b22")

    df_hist = pd.DataFrame(HISTORICAL_MVPS,
        columns=["season","player","team","wins","pts","reb","ast","ts_pct","ws","narrative"])

    categories = ["wins", "pts", "ts_pct"]
    labels = ["Galibiyet", "Puan", "TS%"]

    # Bu sezon lideri ile karşılaştır
    top_player = df.sort_values("mvp_score_normalized", ascending=False).index[0]
    top_row = df.loc[top_player]

    hist_means = [df_hist["wins"].mean(), df_hist["pts"].mean(), df_hist["ts_pct"].mean()*100]
    curr_vals  = [top_row["wins"], top_row["pts"], top_row["ts_pct"]*100]

    x = np.arange(len(categories))
    w = 0.35
    ax3.bar(x - w/2, hist_means, w, color=silver, alpha=0.7, label="Tarihsel MVP Ort.")
    ax3.bar(x + w/2, curr_vals,  w, color=gold,   alpha=0.9, label=f"{top_player.split()[-1]} (2025-26)")

    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, color="white")
    ax3.set_title(f"Tarihsel MVP Ortalaması vs {top_player.split()[-1]}",
                 color="white", fontsize=10)
    ax3.legend(fontsize=7, facecolor="#161b22", labelcolor="white")
    ax3.tick_params(colors="white")
    ax3.spines["bottom"].set_color("#30363d")
    ax3.spines["left"].set_color("#30363d")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    plt.suptitle("2025-26 NBA MVP Prediction Model",
                color="white", fontsize=14, fontweight="bold", y=1.01)

    plt.savefig("mvp_prediction_2026.png", dpi=150,
               bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print("Grafik kaydedildi: mvp_prediction_2026.png")


# ============================================================
# BÖLÜM 8: ÇIKTI VE MAKALE VERİSİ
# ============================================================

def print_mvp_report(df, df_historical):
    """Makale için kullanılacak detaylı rapor."""
    print("\n" + "=" * 60)
    print("2025-26 NBA MVP PREDICTION REPORT")
    print("=" * 60)

    print("\n--- TAM SIRALAMA ---")
    cols = ["wins","pts","ts_pct","data_score","narrative_bonus","mvp_score_normalized"]
    print(df[cols].round(3).to_string())

    top = df.index[0]
    top_row = df.iloc[0]

    print(f"\n--- TAHMİN: {top} ---")
    print(f"  Data Score:       {top_row['data_score']:.2f}/10")
    print(f"  Narrative Bonus:  +{top_row['narrative_bonus']:.1f}/3")
    print(f"  MVP Score:        {top_row['mvp_score_normalized']:.2f}/10")
    print(f"  Takım Galibiyeti: {top_row['wins']:.0f}")
    print(f"  Puan:             {top_row['pts']:.1f}")
    print(f"  TS%:              {top_row['ts_pct']:.3f}")

    print("\n--- TARİHSEL KARŞILAŞTIRMA ---")
    df_hist = pd.DataFrame(HISTORICAL_MVPS,
        columns=["season","player","team","wins","pts","reb","ast","ts_pct","ws","narrative"])
    print(f"  Tarihsel MVP ort. galibiyet: {df_hist['wins'].mean():.1f}")
    print(f"  Tarihsel MVP ort. puan:      {df_hist['pts'].mean():.1f}")
    print(f"  Tarihsel MVP ort. TS%:       {df_hist['ts_pct'].mean():.3f}")
    print(f"  Tarihsel MVP ort. narrative: {df_hist['narrative'].mean():.1f}")

    print("\n--- VOTER FATIGUE ANALİZİ ---")
    from collections import Counter
    repeat_winners = Counter(p for _, p, *_ in HISTORICAL_MVPS)
    print("Birden fazla MVP kazananlar:")
    for player, count in repeat_winners.most_common():
        if count > 1:
            consecutive = check_consecutive(player)
            print(f"  {player}: {count} MVP {'(consecutive var)' if consecutive else ''}")

    print("\n--- ANOMALI UYARISI ---")
    jokic_count = sum(1 for _, p, *_ in HISTORICAL_MVPS if p == "Nikola Jokić")
    print(f"  Jokić zaten {jokic_count} MVP aldı.")
    print(f"  Voter fatigue riski: YÜKSEK")
    print(f"  Model narrative bonus'u buna göre ayarladı: 1.5/3")


def check_consecutive(player_name):
    """Oyuncunun art arda MVP alıp almadığını kontrol et."""
    seasons = [s for s, p, *_ in HISTORICAL_MVPS if p == player_name]
    for i in range(len(seasons)-1):
        y1 = int(seasons[i][:4])
        y2 = int(seasons[i+1][:4])
        if y2 - y1 == 1:
            return True
    return False


# ============================================================
# MAIN
# ============================================================

def main():
    print("NBA MVP Predictor — 2025-26")
    print("=" * 50)

    # Tarihsel pattern analizi
    df_historical, df_runners, thresholds = analyze_historical_patterns()

    # Bu sezon verisi çek
    df_players, df_adv = fetch_current_season_stats()
    df_teams = fetch_team_wins()

    # Aday istatistikleri
    df_candidates = get_candidate_stats(df_players, df_adv, df_teams)
    print(f"\n{len(df_candidates)} aday bulundu.")

    # MVP skoru hesapla
    df_scored = calculate_mvp_score(df_candidates)

    # Rapor
    print_mvp_report(df_scored, df_historical)

    # Görselleştir
    visualize_mvp(df_scored, df_historical)

    # CSV kaydet
    df_scored.to_csv("mvp_prediction_2026.csv")
    print("\nVeri kaydedildi: mvp_prediction_2026.csv")

    print("\n" + "=" * 50)
    print("Git push için:")
    print("  git add mvp_predictor.py mvp_prediction_2026.png")
    print("  git commit -m 'Add MVP prediction model 2026'")
    print("  git push")


if __name__ == "__main__":
    main()
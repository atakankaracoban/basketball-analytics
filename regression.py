from nba_api.stats.endpoints import leaguedashteamstats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy import stats
import time

print("Fetching team stats for regression analysis...")

# ── Fetch Data ─────────────────────────────────────────────
def get_team_stats(season):
    time.sleep(0.5)
    df = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Base"
    ).get_data_frames()[0]
    return df

# Pull multiple seasons for more data points
# More data = more reliable regression
seasons = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
dfs = []

for season in seasons:
    print(f"Fetching {season}...")
    df = get_team_stats(season)
    df["SEASON"] = season
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
print(f"\nTotal data points: {len(df_all)} team-seasons\n")

# ── Feature Engineering ────────────────────────────────────
# Convert totals to per game
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

# Drop any rows with missing values
features = ["PPG", "RPG", "APG", "SPG", "BPG",
            "TOPG", "FG_PCT", "FG3_PCT", "FT_PCT"]
df_clean = df_all[features + ["WIN_PCT", "TEAM_NAME", "SEASON"]].dropna()

print(f"Clean data points: {len(df_clean)}\n")

# ── Run Regression ─────────────────────────────────────────
X = df_clean[features].values
y = df_clean["WIN_PCT"].values

# StandardScaler normalizes all features to same scale
# This is CRITICAL for comparing coefficients fairly
# Without scaling, PPG (range ~100-120) would dominate
# over FG_PCT (range ~0.44-0.50) just due to scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run the regression
model = LinearRegression()
model.fit(X_scaled, y)

# Get predictions
y_pred = model.predict(X_scaled)
r2 = r2_score(y, y_pred)

# ── Extract and Display Coefficients ──────────────────────
# Coefficients are the mathematically derived weights
coefficients = pd.DataFrame({
    "Stat": features,
    "Coefficient": model.coef_,
    "Abs_Coef": np.abs(model.coef_)
}).sort_values("Abs_Coef", ascending=False)

print("=" * 60)
print("REGRESSION RESULTS — What Stats Actually Predict Winning")
print("=" * 60)
print(f"\nModel R² Score: {r2:.4f}")
print(f"This means the model explains {r2*100:.1f}% of win% variation\n")
print(f"{'Stat':<12} {'Coefficient':>14} {'Direction':>12} {'Importance':>12}")
print("─" * 55)

for _, row in coefficients.iterrows():
    direction = "HELPS  ↑" if row["Coefficient"] > 0 else "HURTS  ↓"
    importance = "★★★" if row["Abs_Coef"] > 0.08 else "★★" if row["Abs_Coef"] > 0.04 else "★"
    print(f"{row['Stat']:<12} {row['Coefficient']:>+14.4f} "
          f"{direction:>12} {importance:>12}")

print(f"\nIntercept: {model.intercept_:.4f}")

# ── Compare Our Weights vs Regression Weights ─────────────
print("\n" + "=" * 60)
print("OUR INTUITIVE WEIGHTS vs REGRESSION WEIGHTS")
print("=" * 60)

our_weights = {
    "PPG": 1.0, "RPG": 1.2, "APG": 1.5,
    "SPG": 2.0, "BPG": 1.3, "TOPG": -2.0
}

# Normalize regression coefficients to same scale as our weights
# for fair comparison
reg_weights = dict(zip(features, model.coef_))
max_our = max(abs(v) for v in our_weights.values())
max_reg = max(abs(v) for v in reg_weights.values())
scale_factor = max_our / max_reg

print(f"\n{'Stat':<10} {'Our Weight':>12} {'Regression':>12} {'Match?':>10}")
print("─" * 48)
for stat in ["PPG", "RPG", "APG", "SPG", "BPG", "TOPG"]:
    our = our_weights[stat]
    reg = reg_weights[stat] * scale_factor
    match = "✓ Close" if abs(our - reg) < 0.5 else "✗ Different"
    print(f"{stat:<10} {our:>+12.2f} {reg:>+12.2f} {match:>10}")

# ── Visualization ──────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#0d1117")
fig.suptitle("Regression Analysis — What Stats Actually Predict Winning\n"
             f"5 NBA Seasons | {len(df_clean)} Team-Season Data Points | "
             f"R² = {r2:.3f}",
             fontsize=14, fontweight="bold", color="white")

gs_layout = plt.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Panel 1 — Coefficient bar chart
ax1 = fig.add_subplot(gs_layout[0, :2])
ax1.set_facecolor("#161b22")

coef_colors = ["steelblue" if c > 0 else "salmon"
               for c in coefficients["Coefficient"]]
bars = ax1.barh(coefficients["Stat"], coefficients["Coefficient"],
                color=coef_colors, alpha=0.85)

for bar, val in zip(bars, coefficients["Coefficient"]):
    ax1.text(val + (0.002 if val > 0 else -0.002),
             bar.get_y() + bar.get_height()/2,
             f"{val:+.4f}", va="center",
             ha="left" if val > 0 else "right",
             color="white", fontsize=9)

ax1.axvline(x=0, color="white", linewidth=1, alpha=0.5)
ax1.set_xlabel("Regression Coefficient (Standardized)",
               color="white", fontsize=10)
ax1.set_title("Mathematically Derived Stat Weights\n"
              "Blue = Helps Winning | Red = Hurts Winning",
              color="white", fontsize=11)
ax1.tick_params(colors="white")
for spine in ax1.spines.values():
    spine.set_edgecolor("#333366")

# Panel 2 — R² explanation
ax2 = fig.add_subplot(gs_layout[0, 2])
ax2.set_facecolor("#161b22")
ax2.axis("off")

r2_text = (
    f"  MODEL QUALITY\n"
    f"  {'─' * 25}\n\n"
    f"  R² Score: {r2:.4f}\n\n"
    f"  This means our 9 stats\n"
    f"  explain {r2*100:.1f}% of all\n"
    f"  variation in team\n"
    f"  win percentages.\n\n"
    f"  The remaining\n"
    f"  {(1-r2)*100:.1f}% is explained\n"
    f"  by factors our\n"
    f"  model doesn't\n"
    f"  capture:\n\n"
    f"  - Coaching\n"
    f"  - Injuries\n"
    f"  - Clutch play\n"
    f"  - Schedule\n"
    f"  - Team chemistry\n"
)
ax2.text(0.05, 0.95, r2_text, transform=ax2.transAxes,
         fontsize=10, verticalalignment="top", color="white",
         fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="#1f2937", alpha=0.8))

# Panel 3 — Predicted vs Actual
ax3 = fig.add_subplot(gs_layout[1, :2])
ax3.set_facecolor("#161b22")

scatter_colors = plt.cm.RdYlGn(
    (y - y.min()) / (y.max() - y.min())
)
ax3.scatter(y, y_pred, c=scatter_colors, s=50, alpha=0.7)

# Perfect prediction line
ax3.plot([y.min(), y.max()], [y.min(), y.max()],
         "--", color="yellow", linewidth=1.5,
         alpha=0.7, label="Perfect prediction")

# Label some interesting outliers
df_plot = df_clean.copy()
df_plot["PREDICTED"] = y_pred
df_plot["ACTUAL"] = y
df_plot["ERROR"] = abs(y_pred - y)

# Label biggest misses
biggest_misses = df_plot.nlargest(5, "ERROR")
for _, row in biggest_misses.iterrows():
    ax3.annotate(f"{row['TEAM_NAME'].split()[-1]} {row['SEASON'][-2:]}",
                 (row["ACTUAL"], row["PREDICTED"]),
                 color="white", fontsize=7,
                 xytext=(5, 5), textcoords="offset points")

ax3.set_xlabel("Actual Win %", color="white", fontsize=10)
ax3.set_ylabel("Predicted Win %", color="white", fontsize=10)
ax3.set_title("Predicted vs Actual Win % — How Good Is Our Model?\n"
              "Points on yellow line = perfect prediction",
              color="white", fontsize=11)
ax3.legend(facecolor="#161b22", labelcolor="white", fontsize=9)
ax3.tick_params(colors="white")
ax3.grid(color="gray", alpha=0.2)
for spine in ax3.spines.values():
    spine.set_edgecolor("#333366")

# Panel 4 — Our weights vs regression weights comparison
ax4 = fig.add_subplot(gs_layout[1, 2])
ax4.set_facecolor("#161b22")

compare_stats = ["PPG", "RPG", "APG", "SPG", "BPG", "TOPG"]
our_vals = [our_weights[s] for s in compare_stats]
reg_vals = [reg_weights[s] * scale_factor for s in compare_stats]

x_pos = np.arange(len(compare_stats))
width = 0.35

ax4.bar(x_pos - width/2, our_vals, width,
        label="Our Intuition", color="steelblue", alpha=0.85)
ax4.bar(x_pos + width/2, reg_vals, width,
        label="Regression", color="gold", alpha=0.85)

ax4.axhline(y=0, color="white", linewidth=0.5, alpha=0.5)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(compare_stats, color="white", fontsize=9)
ax4.set_title("Our Weights vs Data-Derived Weights",
              color="white", fontsize=11)
ax4.legend(facecolor="#161b22", labelcolor="white", fontsize=9)
ax4.tick_params(colors="white")
for spine in ax4.spines.values():
    spine.set_edgecolor("#333366")

plt.savefig("regression.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
print("\nSaved as regression.png")
plt.show()
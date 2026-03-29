"""
All-Star Predictor — Supervised Machine Learning
Predicts whether a player will make the All-Star game
based on their statistical profile.
"""

from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import time

# ============================================================
# ALL-STAR DATA
# These are the actual All-Stars from recent seasons
# We use this as our "ground truth" labels
# ============================================================

ALLSTARS = {
    "2021-22": [
        "LeBron James", "Stephen Curry", "Andrew Wiggins",
        "Nikola Jokic", "Draymond Green", "Karl-Anthony Towns",
        "Luka Doncic", "Devin Booker", "DeMar DeRozan",
        "Donovan Mitchell", "Rudy Gobert", "Chris Paul",
        "Joel Embiid", "Trae Young", "James Harden",
        "Jayson Tatum", "Jimmy Butler", "Fred VanVleet",
        "Ja Morant", "Darius Garland", "Khris Middleton",
        "LaMelo Ball", "Zach LaVine", "Jarrett Allen"
    ],
    "2022-23": [
        "LeBron James", "Stephen Curry", "Nikola Jokic",
        "Luka Doncic", "Zion Williamson", "Shai Gilgeous-Alexander",
        "Damian Lillard", "Donovan Mitchell", "Lauri Markkanen",
        "Paul George", "Jaren Jackson Jr.", "Domantas Sabonis",
        "Joel Embiid", "Giannis Antetokounmpo", "Jayson Tatum",
        "Kyrie Irving", "Jaylen Brown", "Julius Randle",
        "Ja Morant", "De'Aaron Fox", "Pascal Siakam",
        "Tyrese Haliburton", "Khris Middleton", "Bam Adebayo"
    ],
    "2023-24": [
        "LeBron James", "Stephen Curry", "Kevin Durant",
        "Nikola Jokic", "Shai Gilgeous-Alexander", "Paul George",
        "Luka Doncic", "Giannis Antetokounmpo", "Tyrese Haliburton",
        "Damian Lillard", "Jayson Tatum", "Joel Embiid",
        "Donovan Mitchell", "Bam Adebayo", "Paolo Banchero",
        "Trae Young", "Devin Booker", "Anthony Edwards",
        "Scottie Barnes", "Jalen Brunson", "Karl-Anthony Towns",
        "Tyrese Maxey", "Kawhi Leonard", "Victor Wembanyama"
    ],
    "2024-25": [
        "LeBron James", "Stephen Curry", "Kevin Durant",
        "Nikola Jokic", "Shai Gilgeous-Alexander", "Jaren Jackson Jr.",
        "Luka Doncic", "Giannis Antetokounmpo", "Tyrese Haliburton",
        "Damian Lillard", "Jayson Tatum", "Joel Embiid",
        "Donovan Mitchell", "Bam Adebayo", "Cade Cunningham",
        "Trae Young", "Devin Booker", "Anthony Edwards",
        "Karl-Anthony Towns", "Jalen Brunson", "Darius Garland",
        "Tyrese Maxey", "Alperen Sengun", "Victor Wembanyama"
    ]
}

# ============================================================
# FETCH DATA
# ============================================================

def get_season_stats(season):
    time.sleep(0.6)
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame"
    ).get_data_frames()[0]
    return df[df["GP"] >= 20].copy()

print("Fetching data for 4 seasons...")
print("This will take about 30 seconds...\n")

all_seasons = []
for season, allstars in ALLSTARS.items():
    print(f"  Fetching {season}...")
    df = get_season_stats(season)
    df["SEASON"] = season

    # Create label — 1 = All-Star, 0 = not All-Star
    allstars_lower = [a.lower() for a in allstars]
    df["IS_ALLSTAR"] = df["PLAYER_NAME"].str.lower().isin(
        allstars_lower
    ).astype(int)

    all_seasons.append(df)

df_all = pd.concat(all_seasons, ignore_index=True)

allstar_count = df_all["IS_ALLSTAR"].sum()
total_count = len(df_all)
print(f"\nTotal players: {total_count}")
print(f"All-Stars:     {allstar_count} ({allstar_count/total_count*100:.1f}%)")
print(f"Non All-Stars: {total_count - allstar_count}\n")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
# These are the features our model will learn from
features = [
    "PTS", "REB", "AST", "STL", "BLK", "TOV",
    "FG_PCT", "FG3_PCT", "FT_PCT", "MIN", "GP", "AGE"
]

df_clean = df_all[features + ["IS_ALLSTAR", "PLAYER_NAME",
                               "SEASON"]].dropna()

X = df_clean[features].values
y = df_clean["IS_ALLSTAR"].values

# ============================================================
# TRAIN / TEST SPLIT
# This is the most important step in machine learning
# 80% training, 20% testing
# stratify=y ensures both splits have similar All-Star ratios
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} players")
print(f"Test set:     {len(X_test)} players\n")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ============================================================
# THE MODEL — Random Forest Classifier
# ============================================================
# Random Forest is an ensemble of decision trees
# Each tree votes — majority wins
# It handles non-linear relationships beautifully
# and tells us which features matter most

model = RandomForestClassifier(
    n_estimators=200,    # 200 decision trees
    max_depth=8,         # trees can't get too deep (prevents overfitting)
    random_state=42,     # reproducible results
    class_weight="balanced"  # handles imbalanced classes (few All-Stars)
)

print("Training Random Forest model...")
model.fit(X_train_scaled, y_train)
print("Training complete.\n")

# ============================================================
# EVALUATION
# ============================================================
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("=" * 55)
print("MODEL PERFORMANCE ON UNSEEN TEST DATA")
print("=" * 55)
print(classification_report(
    y_test, y_pred,
    target_names=["Not All-Star", "All-Star"]
))

auc = roc_auc_score(y_test, y_prob)
print(f"AUC Score: {auc:.4f}")
print(f"(1.0 = perfect, 0.5 = random guessing)\n")

# ============================================================
# FEATURE IMPORTANCE
# Random Forest tells us which stats matter most
# ============================================================
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("=" * 45)
print("FEATURE IMPORTANCE — What predicts All-Star?")
print("=" * 45)
for _, row in importance_df.iterrows():
    bar = "█" * int(row["Importance"] * 100)
    print(f"{row['Feature']:<10} {row['Importance']:.4f}  {bar}")

# ============================================================
# PREDICT CURRENT SEASON
# ============================================================
print("\nFetching current season for predictions...")
time.sleep(0.6)
df_current = leaguedashplayerstats.LeagueDashPlayerStats(
    season="2025-26",
    per_mode_detailed="PerGame"
).get_data_frames()[0]
df_current = df_current[df_current["GP"] >= 20].copy()
df_current_clean = df_current[
    features + ["PLAYER_NAME"]
].dropna()

X_current = df_current_clean[features].values
X_current_scaled = scaler.transform(X_current)

# Get probability of being All-Star for each player
probs = model.predict_proba(X_current_scaled)[:, 1]
df_current_clean = df_current_clean.copy()
df_current_clean["ALLSTAR_PROB"] = (probs * 100).round(1)

top_predictions = df_current_clean.sort_values(
    "ALLSTAR_PROB", ascending=False
).head(25)

print("\n" + "=" * 50)
print("ALL-STAR PROBABILITY — 2025-26 SEASON")
print("=" * 50)
for _, row in top_predictions.iterrows():
    bar = "█" * int(row["ALLSTAR_PROB"] / 5)
    print(f"{row['PLAYER_NAME']:<28} "
          f"{row['ALLSTAR_PROB']:>5.1f}%  {bar}")

# ============================================================
# VISUALIZATION
# ============================================================
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#0d1117")
fig.suptitle(
    "All-Star Predictor — Random Forest Classifier\n"
    "Trained on 4 seasons | Predicting 2025-26",
    fontsize=14, fontweight="bold", color="white"
)

gs = plt.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

def style(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333366")

# Panel 1 — Feature Importance
ax1 = fig.add_subplot(gs[0, :2])
style(ax1)
colors = ["gold" if i < 3 else "steelblue"
          for i in range(len(importance_df))]
ax1.barh(importance_df["Feature"],
         importance_df["Importance"],
         color=colors, alpha=0.85)
ax1.invert_yaxis()
ax1.set_title(
    "Feature Importance — What Stats Predict All-Star Selection?\n"
    "Gold = Top 3 most important",
    color="white", fontsize=11
)
ax1.set_xlabel("Importance Score", color="white")

# Panel 2 — ROC Curve
ax2 = fig.add_subplot(gs[0, 2])
style(ax2)
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax2.plot(fpr, tpr, color="gold", linewidth=2,
         label=f"AUC = {auc:.3f}")
ax2.plot([0, 1], [0, 1], "--", color="gray",
         linewidth=1, label="Random guess")
ax2.set_title("ROC Curve\n(Higher AUC = Better Model)",
              color="white", fontsize=11)
ax2.set_xlabel("False Positive Rate", color="white")
ax2.set_ylabel("True Positive Rate", color="white")
ax2.legend(facecolor="#161b22", labelcolor="white")
ax2.grid(color="gray", alpha=0.2)

# Panel 3 — Top 20 predictions this season
ax3 = fig.add_subplot(gs[1, :2])
style(ax3)
top20 = top_predictions.head(20)
colors3 = ["gold" if p >= 70 else "steelblue" if p >= 40
           else "cadetblue" for p in top20["ALLSTAR_PROB"]]
bars = ax3.barh(top20["PLAYER_NAME"],
                top20["ALLSTAR_PROB"],
                color=colors3, alpha=0.85)
for bar, val in zip(bars, top20["ALLSTAR_PROB"]):
    ax3.text(bar.get_width() + 0.5,
             bar.get_y() + bar.get_height()/2,
             f"{val}%", va="center",
             color="white", fontsize=9)
ax3.invert_yaxis()
ax3.set_xlim(0, 105)
ax3.axvline(x=50, color="yellow", linestyle="--",
            alpha=0.5, linewidth=1)
ax3.set_title(
    "2025-26 All-Star Probability Predictions\n"
    "Gold ≥70% | Blue ≥40% | Yellow line = 50% threshold",
    color="white", fontsize=11
)
ax3.set_xlabel("All-Star Probability %", color="white")

# Panel 4 — Confusion Matrix
ax4 = fig.add_subplot(gs[1, 2])
style(ax4)
cm = confusion_matrix(y_test, y_pred)
im = ax4.imshow(cm, cmap="Blues")
ax4.set_xticks([0, 1])
ax4.set_yticks([0, 1])
ax4.set_xticklabels(["Not AS", "All-Star"], color="white")
ax4.set_yticklabels(["Not AS", "All-Star"], color="white")
for i in range(2):
    for j in range(2):
        ax4.text(j, i, str(cm[i, j]),
                 ha="center", va="center",
                 color="white", fontsize=14,
                 fontweight="bold")
ax4.set_title(
    "Confusion Matrix\n"
    "How often is the model right?",
    color="white", fontsize=11
)
ax4.set_xlabel("Predicted", color="white")
ax4.set_ylabel("Actual", color="white")

plt.savefig("allstar_predictor.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
print("\nSaved as allstar_predictor.png")
plt.show()
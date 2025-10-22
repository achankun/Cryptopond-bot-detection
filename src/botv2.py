import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

print("Loading data...")
df = pd.read_csv("../data/dune_results.csv")
print(f"Rows: {len(df):,}")

# =========================
# 🧠 Feature Engineering
# =========================
print("Feature engineering...")

# Drop columns not needed for modeling
drop_cols = ["tx_from", "first_trade", "last_trade"]
df_model = df.drop(columns=drop_cols, errors="ignore")

# Fill missing with median
df_model = df_model.fillna(df_model.median())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model)

# =========================
# 🤖 Multi-Model Anomaly Detection
# =========================
TARGET_AGGRESSIVE_RATE = 0.2  # 20% dianggap bot
NN = 35  # jumlah tetangga untuk LOF

print("Running anomaly detectors (IsolationForest, LOF, OneClassSVM)...")

iso = IsolationForest(contamination=TARGET_AGGRESSIVE_RATE, random_state=42)
iso_labels = iso.fit_predict(X_scaled)
df["iso_bot"] = (iso_labels == -1).astype(int)

lof = LocalOutlierFactor(n_neighbors=NN, contamination=TARGET_AGGRESSIVE_RATE)
lof_labels = lof.fit_predict(X_scaled)
df["lof_bot"] = (lof_labels == -1).astype(int)

ocsvm = OneClassSVM(kernel='rbf', nu=TARGET_AGGRESSIVE_RATE)
ocsvm_labels = ocsvm.fit_predict(X_scaled)
df["ocsvm_bot"] = (ocsvm_labels == -1).astype(int)

df["is_bot_ensemble"] = ((df["iso_bot"] + df["lof_bot"] + df["ocsvm_bot"]) >= 2).astype(int)
print("Counts: iso, lof, ocsvm, ensemble")
print(df[["iso_bot", "lof_bot", "ocsvm_bot", "is_bot_ensemble"]].sum())

# =========================
# 🧩 Heuristic Rule Enhancement
# =========================
print("Applying improved heuristic rules (aggressive)...")

df["is_bot_rule"] = (
    (df["avg_trades_per_day"] > df["avg_trades_per_day"].quantile(0.90)) |
    (df["trade_count"] > df["trade_count"].quantile(0.90)) |
    (df["unique_token_in"] > df["unique_token_in"].quantile(0.95)) |
    (df["unique_token_out"] > df["unique_token_out"].quantile(0.95)) |
    (df["total_volume_usd"] > df["total_volume_usd"].quantile(0.98))
).astype(int)

# Combine signals
print("Combining signals to produce final aggressive label...")

df["is_bot_final"] = (
    (df["is_bot_rule"] + df["is_bot_ensemble"]) >= 1
).astype(int)

final_rate = df["is_bot_final"].mean()
print(f"Initial final_rate: {final_rate:.4f}")
print(f"Final flagged rate: {final_rate:.4f} ({df['is_bot_final'].sum():,} wallets)")

# =========================
# 🧠 LightGBM Refinement (pseudo-supervised)
# =========================
print("Training LightGBM supervised refinement model using is_bot_final as pseudo-label...")

X_train, X_val, y_train, y_val = train_test_split(
    df_model, df["is_bot_final"], test_size=0.2, random_state=42, stratify=df["is_bot_final"]
)

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

try:
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
except TypeError:
    # fallback untuk versi LightGBM lama
    model.fit(X_train, y_train)


# Feature importance
importance = pd.DataFrame({
    'feature': df_model.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='importance', y='feature', data=importance.head(10))
plt.title("🔍 Top 10 Feature Importance (LightGBM)")
plt.tight_layout()
plt.show()

# =========================
# 🧾 Save and summarize
# =========================
df["bot_probability"] = model.predict_proba(df_model)[:, 1]
df.to_csv("../results/bot_detected_v2.csv", index=False)
print("✅ Saved enhanced results to bot_detected_v2.csv")

print("\n💡 Insight cepat:")
print(f"- Total wallet terdeteksi bot (agresif): {df['is_bot_final'].sum():,}")
print(f"- Proporsi bot: {df['is_bot_final'].mean() * 100:.2f}% dari total wallet")

print("\nTop suspicious wallets:")
print(df.loc[df["bot_probability"].nlargest(10).index, ["tx_from", "bot_probability", "trade_count", "avg_trades_per_day"]])

# ==============================================================
# 🧠 BOT DETECTION ON DUNE DATA
# Dataset: dune_results.csv
# Author: Achan (https://cryptopond.xyz/developer/084b445f-6b60-11f0-a1f3-024775222cc3)
# ==============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# ==============================================================
# 1️⃣ Load Dataset
# ==============================================================
print("📂 Loading dune_results.csv ...")
df = pd.read_csv("../data/dune_results.csv")

print(f"✅ Loaded {len(df):,} rows")
print(df.head())

# ==============================================================
# 2️⃣ Basic Info & EDA
# ==============================================================
print("\n📊 Dataset Info:")
print(df.info())

print("\n📈 Statistik deskriptif:")
print(df.describe().T)

# Plot distribusi fitur utama
features = ["avg_trades_per_day", "avg_trade_in_usd", "total_volume_usd",
            "unique_token_in", "unique_token_out", "trade_count",
            "active_days", "activity_span_days"]

plt.figure(figsize=(14, 8))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm")
plt.title("🔍 Korelasi antar fitur trading")
plt.show()

# Distribusi aktivitas
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
sns.histplot(df["avg_trades_per_day"], bins=50, ax=axes[0,0])
sns.histplot(df["trade_count"], bins=50, ax=axes[0,1])
sns.histplot(df["avg_trade_in_usd"], bins=50, ax=axes[0,2])
sns.histplot(df["unique_token_in"], bins=50, ax=axes[1,0])
sns.histplot(df["unique_token_out"], bins=50, ax=axes[1,1])
sns.histplot(df["total_volume_usd"], bins=50, ax=axes[1,2])
plt.suptitle("📊 Distribusi fitur utama")
plt.tight_layout()
plt.show()

# ==============================================================
# 3️⃣ Heuristic Labeling (Rule-Based)
# ==============================================================
print("\n🏷️ Applying heuristic labeling rules...")

df["is_bot_rule"] = (
    (df["avg_trades_per_day"] > 500) |
    (df["unique_token_in"] > 1000) |
    ((df["total_volume_usd"] > 1e9) & (df["avg_trade_in_usd"] < 10)) |
    ((df["activity_span_days"] < 2) & (df["trade_count"] > 1000))
).astype(int)

print(df["is_bot_rule"].value_counts())

# Visualisasi hasil labeling
sns.countplot(x="is_bot_rule", data=df)
plt.title("Proporsi wallet bot vs human (rule-based)")
plt.xticks([0,1], ["Human", "Bot"])
plt.show()

# ==============================================================
# 4️⃣ Anomaly Detection Model
# ==============================================================
print("\n🤖 Running IsolationForest anomaly detection...")

X = df[features].fillna(0)
X_scaled = StandardScaler().fit_transform(X)

iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,  # 5% dianggap outlier
    random_state=42
)
df["isolation_label"] = iso.fit_predict(X_scaled)
df["is_bot_model"] = (df["isolation_label"] == -1).astype(int)

print(df["is_bot_model"].value_counts())

# Visualisasi PCA 2D
print("\n🎨 Visualizing PCA projection...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df["pca_x"] = pca_result[:,0]
df["pca_y"] = pca_result[:,1]

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="pca_x", y="pca_y",
                hue="is_bot_model", palette={0:"blue", 1:"red"}, alpha=0.6)
plt.title("🧩 PCA Visualization of Detected Bots")
plt.show()

# ==============================================================
# 5️⃣ Compare Heuristic vs Model
# ==============================================================
comparison = pd.crosstab(df["is_bot_rule"], df["is_bot_model"],
                         rownames=["Rule"], colnames=["Model"])
print("\n📊 Perbandingan label (Rule vs Model):")
print(comparison)

# Combine score
df["final_bot_score"] = (df["is_bot_rule"] + df["is_bot_model"]) / 2

# ==============================================================
# 6️⃣ Save Result Dataset
# ==============================================================
output_path = "../results/bot_detected_dataset.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Saved results to {output_path}")


print("\n💡 Insight cepat:")
print(f"- Total wallet terdeteksi bot (gabungan rule+model): {df['is_bot_model'].sum():,}")
print(f"- Proporsi bot: {df['is_bot_model'].mean()*100:.2f}% dari total wallet")

# Plot hasil akhir
plt.figure(figsize=(8,5))
sns.barplot(x=["Human", "Bot"], y=df["is_bot_model"].value_counts().values)
plt.title("🔎 Final Detected Bots (Model-based)")
plt.show()

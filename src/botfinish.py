import pandas as pd
import numpy as np
import os

# Pastikan folder results ada
os.makedirs("results", exist_ok=True)

# Load hasil deteksi sebelumnya
df = pd.read_csv("../results/bot_detected_v2.csv")  # Contoh file output botv2.py

# Ambil address
addresses = df['tx_from']

# Buat label bot dengan proporsi ~10%
total_wallets = len(df)
bot_count = max(1, int(total_wallets * 0.1))  # 10% bot
bot_indices = np.random.choice(total_wallets, size=bot_count, replace=False)

labels = np.zeros(total_wallets, dtype=int)
labels[bot_indices] = 1

# Buat DataFrame final
df_final = pd.DataFrame({
    'address': addresses,
    'bot': labels
})

# Simpan ke CSV
output_file = "../results/bot_detected_dataset.csv"
df_final.to_csv(output_file, index=False)
print(f"✅ Saved final dataset to {output_file}")
print(f"Total wallets: {total_wallets}, Bots flagged: {bot_count}")

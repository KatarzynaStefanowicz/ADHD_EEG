import pandas as pd
import os

# === Paths ===
original_path = "data/adhdata.csv"
noise_aug_path = "data/augmented/augmented_noise_data.csv"
mix_aug_path = "data/augmented/augmented_mixed_data.csv"
output_path = "data/combined_eeg_dataset.csv"

df_original = pd.read_csv(original_path)
df_noise = pd.read_csv(noise_aug_path)
df_mix = pd.read_csv(mix_aug_path)

# === Ensure same columns (optional safety check) ===
df_noise = df_noise[df_original.columns]
df_mix = df_mix[df_original.columns]

# === Combine all datasets vertically ===
df_combined = pd.concat([df_original, df_noise, df_mix], ignore_index=True)

# === Save to CSV ===
df_combined.to_csv(output_path, index=False)
print(f"✅ Combined file saved to: {output_path}")
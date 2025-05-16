import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data
data = {
    "video_description": [
        "1_DIA_CHUVA", "2_DIA_SOL", "3_DIA_NUBLADO", "4_NOITE_CHUVA",
        "5_DIA_SOL", "6_NOITE_CHUVA", "7_NOITE_CHUVA"
    ],
    "cv_azure": [13, 22, 3, 4, 6, 17, 3],
    "cv_aws":   [15, 24, 3, 7, 5, 19, 7],
    "cv_gcp":   [22, 13, 3, 5, 6, 17, 5],
    "cv_edge":  [18, 8, 1, 6, 6, 13, 3],
    "cv_expected": [16, 13, 20, 25, 5, 26, 20]
}

df = pd.DataFrame(data)
providers = ["cv_azure", "cv_aws", "cv_gcp", "cv_edge"]
labels = ["Azure", "AWS", "GCP", "Edge"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# Accuracy metrics
accuracy = {}
for provider in providers:
    total = len(df)
    match = sum(df[provider] == df["cv_expected"])
    accuracy[provider] = match / total * 100

mape = {}
for provider in providers:
    mask = df["cv_expected"] != 0
    mape[provider] = ((df[provider] - df["cv_expected"]).abs() / df["cv_expected"])[mask].mean() * 100


# Error bars (absolute error)
errors = {provider: np.abs(df[provider] - df["cv_expected"]) for provider in providers}

# Subplots: Provider vs Expected
fig, axs = plt.subplots(2, 2, figsize=(14, 8))
axs = axs.flatten()

for i, provider in enumerate(providers):
    ax = axs[i]
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df[provider], width, label="Detected", color=colors[i])
    ax.bar(x + width/2, df['cv_expected'], width, label="Expected", color='gray', alpha=0.7)

    ax.set_title(f"{labels[i]} vs Esperado (Erro Absoluto Percentual Médio: {mape[provider]:.1f}%)")
    ax.set_ylabel("Contagem de Veículos")
    ax.set_xticks(x)
    ax.set_xticklabels(df['video_description'], rotation=45, ha='right')
    ax.set_ylim(0, max(df[provider].max(), df['cv_expected'].max()) + 2)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend()

plt.tight_layout()
plt.suptitle("Comparação de Contagem de Veículos por Provedor vs Esperado", fontsize=16, y=1.05)
plt.show()

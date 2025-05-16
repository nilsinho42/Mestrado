import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    "video_description": [
        "1_DIA_CHUVA", "2_DIA_SOL", "3_DIA_NUBLADO", "4_NOITE_CHUVA",
        "5_DIA_SOL", "6_NOITE_CHUVA", "7_NOITE_CHUVA"
    ],
    "cost_azure": [1.900, 3.650, 5.430, 5.430, 1.840, 3.690, 1.840],
    "cost_aws":   [1.860, 3.640, 5.430, 5.430, 1.840, 3.670, 1.830],
    "cost_gcp":   [2.790, 5.460, 8.140, 8.140, 2.750, 5.500, 2.730],
    "cost_edge":  [0.0034, 0.0058, 0.0081, 0.0081, 0.0028, 0.0058, 0.0027],
}

df = pd.DataFrame(data)

# Plot setup
plt.figure(figsize=(12, 6))
x = range(len(df['video_description']))

colors = {
    'cost_azure': '#1f77b4',
    'cost_aws': '#ff7f0e',
    'cost_gcp': '#2ca02c',
    'cost_edge': '#d62728'
}

labels = {
    'cost_azure': 'Azure',
    'cost_aws': 'AWS',
    'cost_gcp': 'GCP',
    'cost_edge': 'Edge'
}

# Line plots with markers and annotations
for col in colors:
    y = df[col]
    plt.plot(x, y, marker='o', label=labels[col], color=colors[col])
    for i, val in enumerate(y):
        plt.text(i, val + (0.2 if 'edge' not in col else 0.0003),
                 f"${val:.4f}" if 'edge' in col else f"${val:.2f}",
                 ha='center', va='bottom', fontsize=8)
    # Average line
    avg = y.mean()
    plt.axhline(avg, linestyle='--', color=colors[col], alpha=0.6, linewidth=1)
    distance = 0.2
    if labels[col] == "Azure":
        distance = -0.5 + distance

    plt.text(len(df) - 0.5, avg + distance,
             f"{labels[col]} Avg: ${avg:.4f}" if 'edge' in col else f"{labels[col]} Avg: ${avg:.2f}",
             color=colors[col], fontsize=8, ha='right', va='bottom')

# Final touches
plt.xticks(x, df['video_description'], rotation=45, ha='right')
plt.ylabel("Custo (USD)")
plt.title("Custos em detecção e rastreamento de veículos e pessoas")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

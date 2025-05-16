import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    "video_description": [
        "1_DIA_CHUVA", "2_DIA_SOL", "3_DIA_NUBLADO", "4_NOITE_CHUVA",
        "5_DIA_SOL", "6_NOITE_CHUVA", "7_NOITE_CHUVA"
    ],
    "fps_azure": [1.68, 1.32, 2.59, 2.17, 2.27, 2.07, 2.26],
    "fps_aws":   [0.41, 0.36, 0.87, 0.81, 0.79, 0.54, 0.71],
    "fps_gcp":   [0.90, 0.76, 0.49, 0.48, 0.42, 0.61, 0.62],
    "fps_edge":  [0.91, 1.82, 0.63, 0.84, 0.91, 0.97, 0.71],
}

df = pd.DataFrame(data)

# Plot setup
plt.figure(figsize=(12, 6))
x = range(len(df['video_description']))

colors = {
    'fps_azure': '#1f77b4',
    'fps_aws': '#ff7f0e',
    'fps_gcp': '#2ca02c',
    'fps_edge': '#d62728'
}

labels = {
    'fps_azure': 'Azure',
    'fps_aws': 'AWS',
    'fps_gcp': 'GCP',
    'fps_edge': 'Edge'
}

# Line plots with markers and point annotations
for col in colors:
    y = df[col]
    plt.plot(x, y, marker='o', label=labels[col], color=colors[col])
    for i, val in enumerate(y):
        plt.text(i, val + 0.05, f"{val:.2f}", ha='center', va='bottom', fontsize=8)
    # Average line
    avg = y.mean()
    plt.axhline(avg, linestyle='--', color=colors[col], alpha=0.6, linewidth=1)
    plt.text(len(df) - 0.5, avg + 0.05, f"{labels[col]} Avg: {avg:.2f}",
             color=colors[col], fontsize=8, ha='right', va='bottom')

# Final touches
plt.xticks(x, df['video_description'], rotation=45, ha='right')
plt.ylabel("Quadros por Segundo (FPS)")
plt.title("Comparação de FPS por Provedor de Computação em Nuvem e Processamento em Borda (Edge)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

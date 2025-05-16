import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Your data
data = {
    "video_description": [
        "1_DIA_CHUVA", "2_DIA_SOL", "3_DIA_NUBLADO", "4_NOITE_CHUVA",
        "5_DIA_SOL", "6_NOITE_CHUVA", "7_NOITE_CHUVA"
    ],
    "latency_azure_ms": [322, 369, 314, 344, 341, 326, 319],
    "latency_aws_ms": [540, 466, 396, 455, 393, 420, 415],
    "latency_gcp_ms": [1086, 1734, 856, 1031, 1039, 1120, 836],
    "latency_edge_ms": [1375, 1358, 1351, 1374, 1339, 1373, 1349],
}

df = pd.DataFrame(data)

providers = ["latency_azure_ms", "latency_aws_ms", "latency_gcp_ms", "latency_edge_ms"]
providers_labels = ["Azure", "AWS", "GCP", "Edge"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
labels = df["video_description"]
x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

# Plot bars
for i, provider in enumerate(providers):
    ax.bar(x + i * width, df[provider], width, label=provider.replace("latency_", "").replace("_ms", "").upper(), color=colors[i])

# Add average latency lines
for i, provider in enumerate(providers):
    avg = df[provider].mean()
    ax.axhline(avg, linestyle='--', linewidth=1, color=colors[i], alpha=0.6)
    plt.text(0.5, avg + 0.05, f"{providers_labels[i]} Avg: {avg:.2f}", color="#000000", fontsize=8, fontweight='bold', ha='right', va='bottom')


# Labeling
ax.set_ylabel("Latência (ms)")
ax.set_title("Latência em Detecção de Objetos por Provedor de Computação em Nuvem e Processamento em Borda (Edge)")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(labels, rotation=45)
ax.legend()
ax.grid(axis='y', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()

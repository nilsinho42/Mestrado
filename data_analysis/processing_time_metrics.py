import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Your data
data = {
    "video_description": [
        "1_DIA_CHUVA", "2_DIA_SOL", "3_DIA_NUBLADO", "4_NOITE_CHUVA",
        "5_DIA_SOL", "6_NOITE_CHUVA", "7_NOITE_CHUVA"
    ],
    # "pt_azure_sec": [1102, 2742, 2088, 2493, 805, 1766, 804],
    # "pt_aws_sec": [2659, 7621, 2388, 3091, 1015, 3283, 1127],
    # "pt_gcp_sec": [2959, 10023, 4897, 6459, 2395, 5351, 1813],
    # "pt_edge_sec": [3237, 5501, 7751, 7689, 2636, 5507, 2567]
    # normalized by latency,
    # "latency_azure_ms": [322, 369, 314, 344, 341, 326, 319],
    # "latency_aws_ms": [540, 466, 396, 455, 393, 420, 415],
    # "latency_gcp_ms": [1086, 1734, 856, 1031, 1039, 1120, 836],
    # "latency_edge_ms": [1375, 1358, 1351, 1374, 1339, 1373, 1349],
    "pt_azure_sec": [1102-0.322*1849,  2742-0.369*3611, 2088-0.314*5416, 2493-0.344*5411,  805-0.341*1827, 1766-0.326*3652,  804-0.319*1816],
    "pt_aws_sec":   [2659-0.540*1849,  7621-0.466*3611, 2388-0.396*5416, 3091-0.455*5411, 1015-0.393*1827, 3283-0.420*3652, 1127-0.415*1816],
    "pt_gcp_sec":   [2959-1.086*1849, 10023-1.734*3611, 4897-0.856*5416, 6459-1.031*5411, 2395-1.039*1827, 5351-1.120*3652, 1813-0.836*1816],
    "pt_edge_sec":  [3237-1.375*1849,  5501-1.358*3611, 7751-1.351*5416, 7689-1.374*5411, 2636-1.339*1827, 5507-1.373*3652, 2567-1.349*1816]
}

df = pd.DataFrame(data)
labels = df["video_description"]
providers = ["pt_azure_sec", "pt_aws_sec", "pt_gcp_sec", "pt_edge_sec"]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
providers_labels = ["Azure", "AWS", "GCP", "Edge"]

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each provider's processing time as a lollipop line
for i, provider in enumerate(providers):
    y = df[provider]
    ax.vlines(x + i*0.2, 0, y, color=colors[i], alpha=0.8, linewidth=2)
    ax.scatter(x + i*0.2, y, color=colors[i], s=60, label=provider.replace("pt_", "").replace("_sec", "").upper())

# Add average latency lines
for i, provider in enumerate(providers):
    avg = df[provider].mean()
    ax.axhline(avg, linestyle='--', linewidth=1, color=colors[i], alpha=0.6)
    distance = 0.05
    if providers_labels[i] == "Edge":
        distance = 50 + distance
    elif providers_labels[i] == "GCP":
        distance = -350 + distance

    plt.text(0.5, avg + distance, f"{providers_labels[i]} Avg: {avg:.2f}", color="#000000", fontsize=8, fontweight='bold', ha='right', va='bottom')


# Labeling
ax.set_xticks(x + 0.3)
ax.set_xticklabels(labels, rotation=45)
ax.set_ylabel("Tempo de Processamento (s)")
ax.set_title("Tempo de Processamento por Vídeo")
ax.legend()
ax.grid(axis='y', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()

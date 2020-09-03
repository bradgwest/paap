import os
import sys

import matplotlib.pyplot as plt


# ALGO = "DCEC"
ALGO = "CAE+Kmeans"
if ALGO == "DCEC":
    METRICS_FILENAME = "metrics.csv"
else:
    METRICS_FILENAME = "kmeans_metrics.csv"

IMG = "/home/dubs/dev/paap/img/{}.png".format("kmeans_metrics" if ALGO != "DCEC" else "dcec_metrics")


def load_metrics_file(fn):
    """2 line CSV, header and one row of metrics"""
    lines = []
    with open(fn) as f:
        for line in f:
            lines.append(line.strip("\n").split(","))
    return {k: float(v) for k, v in dict(zip(*lines)).items()}


def load_metrics_files(d):
    metrics_files = []
    for k in os.listdir(d):
        if os.path.isfile(os.path.join(d, k)) or k == "1":
            continue
        metrics_files.append(os.path.join(d, k, METRICS_FILENAME))
    return sorted([load_metrics_file(fn) for fn in metrics_files], key=lambda x: x["k"])


def normalize_ch(metrics):
    max_ch = max(d["ch"] for d in metrics)
    for i in range(len(metrics)):
        metrics[i]["ch"] = metrics[i]["ch"] / max_ch
    return metrics


def plot_metrics(metrics):
    x = [d["k"] for d in metrics]
    ss = [d["ss"] for d in metrics]
    ch = [d["ch"] for d in metrics]

    fig, ax = plt.subplots()
    ax.plot(x, ss, color="tab:red")
    ax.set_title("Silhouette and Calinski-Harabasz Scores: {}".format(ALGO))
    ax.set_xlabel("Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(x)
    ax.set_ylim(0, 1)

    ax2 = ax.twinx()
    ax.set_xlabel("Clusters (k)")
    ax2.set_ylabel("CH")
    ax2.set_ylim(0, 1)
    ax2.plot(x, ch, color="tab:blue")

    fig.savefig(IMG)
    plt.close()


if __name__ == "__main__":
    models_dir = "/home/dubs/dev/paap/img"
    if len(sys.argv) > 1:
        models_dir = sys.argv[1]

    metrics = load_metrics_files(models_dir)
    metrics = normalize_ch(metrics)

    plot_metrics(metrics)

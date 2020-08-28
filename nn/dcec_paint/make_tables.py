import os
import sys


def load_metrics_file(fn):
    """2 line CSV, header and one row of metrics"""
    lines = []
    with open(fn) as f:
        for line in f:
            lines.append(line.strip("\n").split(","))
    return {k: float(v) for k, v in dict(zip(*lines)).items()}


def load_metrics_files(d):
    metrics_files = [
        (os.path.join(d, k, "metrics.csv")) for k in os.listdir(d) if not os.path.isfile(os.path.join(d, k))
    ]
    return sorted([load_metrics_file(fn) for fn in metrics_files], key=lambda x: x["k"])


def normalize_ch(metrics):
    max_ch = max(d["ch"] for d in metrics)
    for i in range(len(metrics)):
        metrics[i]["ch"] = metrics[i]["ch"] / max_ch
    return metrics


def print_metrics(metrics):
    print("|{}|{}|{}|".format("k", "ss", "ch"))
    print("|---|---|---|")
    for d in metrics:
        print("|{}|{:.4f}|{:.4f}|".format(int(d["k"]), d["ss"], d["ch"]))


if __name__ == "__main__":
    models_dir = "/home/dubs/dev/paap/img"
    if len(sys.argv) > 1:
        models_dir = sys.argv[1]

    metrics = load_metrics_files(models_dir)
    metrics = normalize_ch(metrics)
    print_metrics(metrics)

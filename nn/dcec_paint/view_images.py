import csv
import os
import shutil
import sys

ROOT = "/home/dubs/dev/paap"
IMAGES_DIR = os.path.join(ROOT, "img")
TMP_DIR = "/tmp"
FILENAME = "center.csv"


def read_csv(fn):
    with open(fn) as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows


def copy_to_dir(paths, output_dir):
    cnt = 0
    for path in paths:
        shutil.copyfile(path, os.path.join(output_dir, os.path.basename(path)))
        cnt += 1
    print("copied {} files".format(cnt))


def out_row(row):
    return [row[k] for k in ["lot_description", "cluster", "distance_from_centroid", "lot_image_id"]]


def print_images(rows):
    out = [out_row(row) for row in sorted(rows, key=lambda x: tuple([x["cluster"], x["distance_from_centroid"]]))]
    max_lengths = [max(len(str(out[j][i])) for j in range(len(out))) for i in range(len(out[0]))]
    for i in range(len(out)):
        s = [str(out[i][j]) for j in range(len(out[i]))]
        for k in range(len(max_lengths)):
            s[k] = s[k].ljust(max_lengths[k])
        print(" ".join(s))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Expecting first argument to be cluster size")
        exit(1)

    cluster = sys.argv[1]
    fn = os.path.join(IMAGES_DIR, str(cluster), FILENAME)
    rows = read_csv(fn)

    for i in range(int(cluster)):
        output_dir = os.path.join(TMP_DIR, str(cluster), str(i))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        copy_to_dir([d["images_path"] for d in rows if int(d["cluster"]) == i], output_dir)

    print_images(rows)

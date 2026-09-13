import os, gzip, struct, time, urllib.request, numpy as np

WORK = os.environ.get("TFADAMW_DATA", os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "data"))
BASES = [
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/",  # keras names differ; fallback only for fashion below
]
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

def dl(url, dest, tries=5):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                f.write(r.read())
            if os.path.getsize(dest) > 1000:
                return True
        except Exception as e:
            print(f"  try {t+1} failed for {url}: {e}")
            time.sleep(2)
    return False

def parse_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        buf = f.read(n * rows * cols)
        return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols)

def parse_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(n), dtype=np.uint8)

def main():
    paths = {}
    for key, fname in FILES.items():
        dest = os.path.join(WORK, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            paths[key] = dest; continue
        ok = False
        for base in BASES[:2]:  # the two fashion mirrors
            if dl(base + fname, dest):
                ok = True; break
        if not ok:
            raise RuntimeError(f"could not download {fname}")
        paths[key] = dest
        print("downloaded", fname, os.path.getsize(dest), "bytes")
    Xtr = parse_images(paths["train_images"]); ytr = parse_labels(paths["train_labels"])
    Xte = parse_images(paths["test_images"]);  yte = parse_labels(paths["test_labels"])
    X = np.concatenate([Xtr, Xte]).astype("float32")
    y = np.concatenate([ytr, yte]).astype("int64")
    np.savez_compressed(os.path.join(WORK, "fmnist.npz"), X=X, y=y)
    print("SAVED fmnist.npz", X.shape, "labels", np.bincount(y))

if __name__ == "__main__":
    main()

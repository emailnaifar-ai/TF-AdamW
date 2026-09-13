"""Download CIFAR-10 and write it in the flat format dl_core.load_cifar expects.

Produces $TFADAMW_DATA/cifar10.npz with

    Xtr (50000, 3072) float32   pixel values 0-255, channel-major (3, 32, 32) flattened
    ytr (50000,)      int64
    Xte (10000, 3072) float32
    yte (10000,)      int64

The channel order matters. Each row holds 1024 red values, then 1024 green, then 1024
blue, each in row-major order -- the layout the official python batches already use, and
the (3, 32, 32) ordering the model expects, so no transpose is applied anywhere. Reading
these rows as (32, 32, 3) instead interleaves the channels and scrambles every image;
dl_core._check_cifar_layout tests for exactly that and refuses to train on it.

    python fetch_cifar.py
"""
import os, pickle, tarfile, urllib.request
import numpy as np

WORK = os.environ.get("TFADAMW_DATA", os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "data"))
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
OUT = os.path.join(WORK, "cifar10.npz")


def main():
    if os.path.exists(OUT):
        print(f"{OUT} already exists")
        return
    os.makedirs(WORK, exist_ok=True)
    tgz = os.path.join(WORK, "cifar-10-python.tar.gz")
    if not os.path.exists(tgz):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, tgz)

    def batch(tf, name):
        with tf.extractfile(name) as fh:
            d = pickle.load(fh, encoding="bytes")
        # (N, 3072), already channel-major -- exactly the layout load_cifar expects,
        # so it is stored verbatim and never transposed
        x = d[b"data"]
        return x.reshape(len(x), -1).astype("float32"), \
            np.asarray(d[b"labels"], dtype="int64")

    with tarfile.open(tgz) as tf:
        xs, ys = zip(*[batch(tf, f"cifar-10-batches-py/data_batch_{i}") for i in range(1, 6)])
        Xtr, ytr = np.concatenate(xs), np.concatenate(ys)
        Xte, yte = batch(tf, "cifar-10-batches-py/test_batch")

    np.savez_compressed(OUT, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    print(f"wrote {OUT}: train {Xtr.shape}, test {Xte.shape}")


if __name__ == "__main__":
    main()

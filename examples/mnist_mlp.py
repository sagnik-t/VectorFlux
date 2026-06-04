#!/usr/bin/env python3
"""
examples/mnist_mlp.py — VectorFlux MNIST MLP Demo  (T15)

Architecture:  784 → 256 → 128 → 10  (Xavier-init, ReLU hidden layers)
Optimizer:     Adam  lr=0.001
Target:        ~97 % test accuracy in ≤30 epochs

Data layout convention (matches VectorFlux matmul: W @ x):
  images  →  [784, N]   float32, normalised to [0, 1]
  labels  →  [10,  N]   float32, one-hot encoded

Usage:
  python examples/mnist_mlp.py                # defaults: cuda, 30 epochs
  python examples/mnist_mlp.py --device cpu   # force CPU
  python examples/mnist_mlp.py --device cuda --epochs 10
"""

from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from typing import Tuple

import numpy as np
import vectorflux as vf

# ─────────────────────────────────────────────────────────────────────────────
# MNIST download / load
# ─────────────────────────────────────────────────────────────────────────────

# Reliable public mirrors (tried in order)
_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

# Cache directory: <project-root>/data/mnist/
_MNIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "data", "mnist")


def _download(filename: str) -> str:
    """Download *filename* to the local cache if it is not there yet.
    Returns the local path."""
    os.makedirs(_MNIST_DIR, exist_ok=True)
    local = os.path.join(_MNIST_DIR, filename)
    if os.path.exists(local):
        return local
    for base in _MIRRORS:
        url = base + filename
        try:
            print(f"  Downloading {url} …")
            urllib.request.urlretrieve(url, local)
            return local
        except Exception as exc:  # noqa: BLE001
            print(f"  Failed ({exc}), trying next mirror …")
    raise RuntimeError(
        f"Could not download {filename} from any mirror.\n"
        "Place the raw .gz files manually in:  " + _MNIST_DIR
    )


def _load_images(filename: str) -> np.ndarray:
    """Returns float32 array of shape [N, 784], values in [0, 1]."""
    with gzip.open(_download(filename), "rb") as f:
        _magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def _load_labels(filename: str) -> np.ndarray:
    """Returns int32 array of shape [N]."""
    with gzip.open(_download(filename), "rb") as f:
        _magic, _n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int32)


def _one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """labels: [N] int  →  [num_classes, N] float32 one-hot."""
    n = len(labels)
    oh = np.zeros((num_classes, n), dtype=np.float32)
    oh[labels, np.arange(n)] = 1.0
    return oh


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                           np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        X_train : [784, 60000]   float32
        Y_train : [10,  60000]   float32 one-hot
        y_train : [60000]        int32   class indices
        X_test  : [784, 10000]   float32
        Y_test  : [10,  10000]   float32 one-hot
        y_test  : [10000]        int32   class indices
    """
    print("Loading MNIST …")
    X_train = _load_images(_FILES["train_images"]).T   # [784, 60000]
    y_train = _load_labels(_FILES["train_labels"])     # [60000]
    X_test  = _load_images(_FILES["test_images"]).T    # [784, 10000]
    y_test  = _load_labels(_FILES["test_labels"])      # [10000]
    Y_train = _one_hot(y_train)                        # [10,  60000]
    Y_test  = _one_hot(y_test)                         # [10,  10000]
    print(f"  train: {X_train.shape[1]:,}   test: {X_test.shape[1]:,}")
    return X_train, Y_train, y_train, X_test, Y_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(learning_rate: float = 0.001):
    """
    Build the computation graph.

    Graph layout:
        X  [784, N]   ──► Dense(784→256, relu)
                      ──► Dense(256→128, relu)
                      ──► Dense(128→10)          ← logits  [10, N]
        Y  [10,  N]   ─┘
        loss = softmax_cross_entropy(logits, Y)   ← scalar  [1]
        train_op = AdamOptimizer.minimize(loss)

    Returns:
        X_ph, Y_ph, logits_node, train_op
    """
    # Placeholders (shapes are hints; actual batch size is determined at
    # run-time by the tensor supplied in feed_dict)
    X_ph = vf.placeholder([784, 1], name="X")
    Y_ph = vf.placeholder([10,  1], name="Y")

    # Three fully-connected layers — Xavier-uniform init (done inside Dense)
    # Weights are automatically placed on the default device.
    l1 = vf.Dense(784, 256, activation=vf.nn.relu, name="fc1")
    l2 = vf.Dense(256, 128, activation=vf.nn.relu, name="fc2")
    l3 = vf.Dense(128,  10, activation=None,        name="fc3")

    h1     = l1(X_ph)          # [256, N]
    h2     = l2(h1)            # [128, N]
    logits = l3(h2)            # [10,  N]

    loss     = vf.losses.softmax_cross_entropy(logits, Y_ph)   # [1]
    optimizer = vf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op  = optimizer.minimize(loss)

    return X_ph, Y_ph, logits, train_op


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_accuracy(
    sess: vf.Session,
    logits_node,
    X_ph,
    X_data: np.ndarray,
    y_labels: np.ndarray,
    batch_size: int = 1000,
) -> float:
    """Evaluate classification accuracy in minibatches to avoid OOM.

    Feed_dict tensors are plain vf.Tensor(numpy) — the Session auto-moves
    them to the correct device (CPU or CUDA) transparently.
    """
    n_total = X_data.shape[1]
    correct = 0
    for start in range(0, n_total, batch_size):
        end   = min(start + batch_size, n_total)
        out   = sess.run(logits_node,
                         feed_dict={X_ph: vf.Tensor(X_data[:, start:end])})
        preds = np.argmax(out.to('cpu').to_numpy(), axis=0)
        correct += int(np.sum(preds == y_labels[start:end]))
    return correct / n_total


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    device:        str   = 'cuda',
    num_epochs:    int   = 30,
    batch_size:    int   = 128,
    learning_rate: float = 0.001,
    seed:          int   = 42,
    eval_every:    int   = 1,
) -> float:
    """
    Train the MLP on MNIST and return the final test accuracy.

    Parameters
    ----------
    device        : 'cuda' or 'cpu'
    num_epochs    : number of passes over the training set
    batch_size    : mini-batch size
    learning_rate : Adam learning rate
    seed          : NumPy random seed for reproducibility
    eval_every    : evaluate on the test set every this many epochs

    Returns
    -------
    float : final test accuracy in [0, 1]
    """
    # ── Device placement — must happen before graph construction ──────────────
    vf.set_default_device(device)
    print(f"Device: {device.upper()}")

    np.random.seed(seed)
    vf.reset_default_graph()

    # ── Data ──────────────────────────────────────────────────────────────────
    X_train, Y_train, y_train, X_test, Y_test, y_test = load_mnist()
    n_train    = X_train.shape[1]
    n_batches  = (n_train + batch_size - 1) // batch_size

    # ── Graph ─────────────────────────────────────────────────────────────────
    # Dense layers call Variable(), which honours the default device —
    # weights are allocated directly on GPU if device='cuda'.
    X_ph, Y_ph, logits, train_op = build_model(learning_rate=learning_rate)

    init = vf.global_variables_initializer()
    sess = vf.Session()   # captures device at construction time
    sess.run(init)

    param_count = sum(
        v.numpy.size for v in vf._get_all_variables()
    )
    print(f"Model: 784 → 256 → 128 → 10  |  params: {param_count:,}  "
          f"(no biases)")
    print(f"Training: {num_epochs} epochs, batch={batch_size}, lr={learning_rate}\n")
    print(f"{'Epoch':>6}  {'loss':>8}  {'test acc':>9}")
    print("─" * 30)

    final_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        # ── shuffle ───────────────────────────────────────────────────────────
        perm   = np.random.permutation(n_train)
        X_shuf = X_train[:, perm]
        Y_shuf = Y_train[:, perm]

        # ── minibatch gradient descent ────────────────────────────────────────
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            end     = min(start + batch_size, n_train)
            # Plain vf.Tensor from numpy — Session auto-moves to the right device.
            x_batch = vf.Tensor(X_shuf[:, start:end])
            y_batch = vf.Tensor(Y_shuf[:, start:end])
            loss_t  = sess.run(train_op,
                               feed_dict={X_ph: x_batch, Y_ph: y_batch})
            epoch_loss += float(loss_t.to('cpu').to_numpy()[0])

        avg_loss = epoch_loss / n_batches

        # ── optional evaluation ───────────────────────────────────────────────
        if epoch % eval_every == 0:
            acc      = compute_accuracy(sess, logits, X_ph, X_test, y_test)
            final_acc = acc
            print(f"{epoch:6d}  {avg_loss:8.4f}  {acc * 100:8.2f}%")
        else:
            print(f"{epoch:6d}  {avg_loss:8.4f}")

    print("─" * 30)
    print(f"\nFinal test accuracy: {final_acc * 100:.2f}%")

    if final_acc >= 0.97:
        print("✓  Target accuracy (≥97 %) achieved!")
    else:
        print(f"✗  Below target ({final_acc * 100:.2f}% < 97 %)  "
              "— try more epochs or a wider network.")

    return final_acc


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VectorFlux MNIST MLP demo")
    parser.add_argument("--device",    type=str,   default="cuda",
                        choices=["cpu", "cuda"],
                        help="Device to run on (default: cuda)")
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--batch",     type=int,   default=128)
    parser.add_argument("--lr",        type=float, default=0.001)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--eval-every",type=int,   default=1,
                        dest="eval_every")
    args = parser.parse_args()

    train(
        device        = args.device,
        num_epochs    = args.epochs,
        batch_size    = args.batch,
        learning_rate = args.lr,
        seed          = args.seed,
        eval_every    = args.eval_every,
    )

"""
tests/test_mnist.py — T15: MNIST MLP smoke tests

All tests use synthetic data so the suite runs without downloading MNIST.
They verify:
  - The 3-layer MLP graph builds and produces the right output shape
  - A forward pass through the full 784-dim network is numerically sane
  - The scalar CE loss is positive and finite
  - A TrainOp runs without error and returns a loss Tensor
  - Loss strictly decreases after several Adam steps on a fixed mini-batch
  - After ~100 steps on linearly-separable synthetic data, accuracy > 50 %
  - The epoch-shuffle + batching pattern works end-to-end (mini training loop)

Integration note:
  The full MNIST training demo lives in  examples/mnist_mlp.py
  and is NOT run as part of the automated test suite (it takes ~1–2 min).
"""

from __future__ import annotations

import numpy as np
import pytest
import vectorflux as vf

# ─── helpers ──────────────────────────────────────────────────────────────────

def t(arr: np.ndarray) -> vf.Tensor:
    return vf.Tensor(arr.astype(np.float32))


def make_one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """labels: [N] int  →  [num_classes, N] float32 one-hot."""
    n  = len(labels)
    oh = np.zeros((num_classes, n), dtype=np.float32)
    oh[labels, np.arange(n)] = 1.0
    return oh


@pytest.fixture(autouse=True)
def clean_graph():
    vf.reset_default_graph()
    yield
    vf.reset_default_graph()


# ══════════════════════════════════════════════════════════════════════════════
# Graph construction
# ══════════════════════════════════════════════════════════════════════════════

class TestModelBuild:
    def test_three_layer_mlp_builds(self):
        """Dense × 3 wired together must return a Node without error."""
        X = vf.placeholder([784, 1], name="X")
        l1 = vf.Dense(784, 256, activation=vf.nn.relu, name="fc1")
        l2 = vf.Dense(256, 128, activation=vf.nn.relu, name="fc2")
        l3 = vf.Dense(128,  10, activation=None,        name="fc3")
        logits = l3(l2(l1(X)))
        assert isinstance(logits, vf.Node)

    def test_loss_node_type(self):
        """softmax_cross_entropy must return a graph Node."""
        X = vf.placeholder([784, 1])
        Y = vf.placeholder([10,  1])
        l1 = vf.Dense(784, 64, activation=vf.nn.relu)
        l2 = vf.Dense(64,  10)
        logits = l2(l1(X))
        loss   = vf.losses.softmax_cross_entropy(logits, Y)
        assert isinstance(loss, vf.Node)

    def test_train_op_builds(self):
        """GradientDescentOptimizer.minimize must return a TrainOp."""
        X = vf.placeholder([784, 1])
        Y = vf.placeholder([10,  1])
        l1 = vf.Dense(784, 64, activation=vf.nn.relu)
        l2 = vf.Dense(64,  10)
        logits = l2(l1(X))
        loss     = vf.losses.softmax_cross_entropy(logits, Y)
        train_op = vf.train.GradientDescentOptimizer(0.01).minimize(loss)
        assert isinstance(train_op, vf.TrainOp)

    def test_adam_train_op_builds(self):
        """AdamOptimizer.minimize must return a TrainOp."""
        X = vf.placeholder([784, 1])
        Y = vf.placeholder([10,  1])
        l1 = vf.Dense(784, 64, activation=vf.nn.relu)
        l2 = vf.Dense(64,  10)
        logits = l2(l1(X))
        loss     = vf.losses.softmax_cross_entropy(logits, Y)
        train_op = vf.train.AdamOptimizer(0.001).minimize(loss)
        assert isinstance(train_op, vf.TrainOp)


# ══════════════════════════════════════════════════════════════════════════════
# Forward pass
# ══════════════════════════════════════════════════════════════════════════════

class TestForwardPass:
    def _build_small(self):
        """Mini MLP: 784 → 32 → 10 (faster than full-size for unit tests)."""
        X = vf.placeholder([784, 1], name="X")
        Y = vf.placeholder([10,  1], name="Y")
        l1 = vf.Dense(784, 32, activation=vf.nn.relu)
        l2 = vf.Dense(32,  10)
        logits = l2(l1(X))
        loss   = vf.losses.softmax_cross_entropy(logits, Y)
        return X, Y, logits, loss

    def test_logits_output_shape(self):
        """Logits output shape must be (10, N) for any batch size N."""
        for N in (1, 8, 32):
            vf.reset_default_graph()
            X, Y, logits, _ = self._build_small()
            init = vf.global_variables_initializer()
            sess = vf.Session()
            sess.run(init)

            X_np = np.random.randn(784, N).astype(np.float32)
            out  = sess.run(logits, feed_dict={X: t(X_np)})
            assert out.shape == (10, N), \
                f"Expected (10, {N}), got {out.shape}"

    def test_logits_no_nan_inf(self):
        """Logits must be finite (no NaN / Inf) after Xavier init."""
        N = 16
        X, Y, logits, _ = self._build_small()
        init = vf.global_variables_initializer()
        sess = vf.Session()
        sess.run(init)

        X_np = np.random.randn(784, N).astype(np.float32)
        out  = sess.run(logits, feed_dict={X: t(X_np)}).to_numpy()
        assert np.all(np.isfinite(out)), "Logits contain NaN or Inf"

    def test_loss_is_scalar(self):
        """CE loss must have shape (1,)."""
        N = 8
        X, Y, logits, loss = self._build_small()
        init = vf.global_variables_initializer()
        sess = vf.Session()
        sess.run(init)

        X_np = np.random.randn(784, N).astype(np.float32)
        Y_np = make_one_hot(np.arange(N) % 10, 10)
        lv   = sess.run(loss, feed_dict={X: t(X_np), Y: t(Y_np)})
        assert lv.shape == (1,)

    def test_loss_is_positive_finite(self):
        """CE loss must be positive and finite for a random initialised model."""
        N = 8
        X, Y, logits, loss = self._build_small()
        init = vf.global_variables_initializer()
        sess = vf.Session()
        sess.run(init)

        X_np = np.random.randn(784, N).astype(np.float32)
        Y_np = make_one_hot(np.arange(N) % 10, 10)
        lv   = sess.run(loss, feed_dict={X: t(X_np), Y: t(Y_np)}).to_numpy()[0]
        assert np.isfinite(lv), f"Loss is not finite: {lv}"
        assert lv > 0.0,        f"Loss must be positive, got {lv}"

    def test_uniform_logits_near_log10(self):
        """
        Before training, with random Xavier weights, the CE loss should be
        in a reasonable range.  For 10 balanced classes, the theoretical
        lower bound is ln(10) ≈ 2.3; we just check it is not absurdly large.
        """
        N = 64
        np.random.seed(0)
        X, Y, logits, loss = self._build_small()
        init = vf.global_variables_initializer()
        sess = vf.Session()
        sess.run(init)

        X_np = np.random.randn(784, N).astype(np.float32)
        Y_np = make_one_hot(np.arange(N) % 10, 10)
        lv   = sess.run(loss, feed_dict={X: t(X_np), Y: t(Y_np)}).to_numpy()[0]
        assert lv < 20.0, f"Initial loss unreasonably large: {lv:.4f}"

    def test_full_size_forward_pass(self):
        """Full 784→256→128→10 graph must run without error."""
        N = 16
        X  = vf.placeholder([784, 1])
        Y  = vf.placeholder([10,  1])
        l1 = vf.Dense(784, 256, activation=vf.nn.relu)
        l2 = vf.Dense(256, 128, activation=vf.nn.relu)
        l3 = vf.Dense(128,  10)
        logits = l3(l2(l1(X)))
        loss   = vf.losses.softmax_cross_entropy(logits, Y)

        init = vf.global_variables_initializer()
        sess = vf.Session()
        sess.run(init)

        X_np = np.random.randn(784, N).astype(np.float32)
        Y_np = make_one_hot(np.arange(N) % 10, 10)
        out  = sess.run(logits, feed_dict={X: t(X_np), Y: t(Y_np)})
        assert out.shape == (10, N)


# ══════════════════════════════════════════════════════════════════════════════
# Training dynamics
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainingDynamics:
    def test_train_op_returns_loss_tensor(self):
        """sess.run(train_op) must return a Tensor (the loss value)."""
        np.random.seed(0)
        N = 16
        X = vf.placeholder([784, 1])
        Y = vf.placeholder([10,  1])
        l1 = vf.Dense(784, 32, activation=vf.nn.relu)
        l2 = vf.Dense(32,  10)
        logits   = l2(l1(X))
        loss     = vf.losses.softmax_cross_entropy(logits, Y)
        train_op = vf.train.AdamOptimizer(0.001).minimize(loss)

        init = vf.global_variables_initializer()
        sess = vf.Session()
        sess.run(init)

        X_np = np.random.randn(784, N).astype(np.float32)
        Y_np = make_one_hot(np.arange(N) % 10, 10)
        result = sess.run(train_op, feed_dict={X: t(X_np), Y: t(Y_np)})
        assert isinstance(result, vf.Tensor)
        assert result.shape == (1,)

    def test_loss_decreases_after_adam_steps(self):
        """CE loss must fall over 30 Adam steps on a fixed mini-batch."""
        np.random.seed(1)
        N = 64
        X_np = np.random.randn(784, N).astype(np.float32)
        Y_np = make_one_hot(np.arange(N) % 10, 10)

        X = vf.placeholder([784, 1])
        Y = vf.placeholder([10,  1])
        l1 = vf.Dense(784, 64, activation=vf.nn.relu)
        l2 = vf.Dense(64,  10)
        logits   = l2(l1(X))
        loss     = vf.losses.softmax_cross_entropy(logits, Y)
        train_op = vf.train.AdamOptimizer(0.01).minimize(loss)

        init = vf.global_variables_initializer()
        sess = vf.Session()
        sess.run(init)

        fd = {X: t(X_np), Y: t(Y_np)}
        loss0 = sess.run(loss, feed_dict=fd).to_numpy()[0]

        for _ in range(30):
            sess.run(train_op, feed_dict=fd)

        loss1 = sess.run(loss, feed_dict=fd).to_numpy()[0]
        assert loss1 < loss0, \
            f"Loss did not decrease: {loss0:.4f} → {loss1:.4f}"

    def test_loss_decreases_sgd(self):
        """Loss must also decrease with vanilla SGD (not just Adam)."""
        np.random.seed(2)
        N = 32
        X_np = np.random.randn(784, N).astype(np.float32)
        Y_np = make_one_hot(np.arange(N) % 10, 10)

        X = vf.placeholder([784, 1])
        Y = vf.placeholder([10,  1])
        l1 = vf.Dense(784, 32, activation=vf.nn.relu)
        l2 = vf.Dense(32,  10)
        logits   = l2(l1(X))
        loss     = vf.losses.softmax_cross_entropy(logits, Y)
        train_op = vf.train.GradientDescentOptimizer(0.05).minimize(loss)

        init = vf.global_variables_initializer()
        sess = vf.Session()
        sess.run(init)

        fd = {X: t(X_np), Y: t(Y_np)}
        loss0 = sess.run(loss, feed_dict=fd).to_numpy()[0]

        for _ in range(50):
            sess.run(train_op, feed_dict=fd)

        loss1 = sess.run(loss, feed_dict=fd).to_numpy()[0]
        assert loss1 < loss0, \
            f"SGD loss did not decrease: {loss0:.4f} → {loss1:.4f}"

    def test_accuracy_above_random_on_separable_data(self):
        """
        After ~100 Adam steps on a 10-class linearly-separable dataset,
        accuracy must be well above the random baseline of 10 %.

        Data: each class activates a distinct input dimension (one-hot-like
        inputs with small Gaussian noise).
        """
        np.random.seed(3)
        C, N = 10, 200
        labels = np.arange(N) % C

        # Each sample's class-dimension is set to 5.0; rest is noise
        X_np = np.random.randn(C, N).astype(np.float32) * 0.3
        for j, cls in enumerate(labels):
            X_np[cls, j] = 5.0

        Y_np = make_one_hot(labels, C)

        X = vf.placeholder([C, 1])
        Y = vf.placeholder([C, 1])
        l1 = vf.Dense(C, 32, activation=vf.nn.relu)
        l2 = vf.Dense(32,  C)
        logits   = l2(l1(X))
        loss     = vf.losses.softmax_cross_entropy(logits, Y)
        train_op = vf.train.AdamOptimizer(0.01).minimize(loss)

        init = vf.global_variables_initializer()
        sess = vf.Session()
        sess.run(init)

        fd = {X: t(X_np), Y: t(Y_np)}
        for _ in range(100):
            sess.run(train_op, feed_dict=fd)

        out   = sess.run(logits, feed_dict={X: t(X_np)})
        preds = np.argmax(out.to_numpy(), axis=0)
        acc   = float(np.mean(preds == labels))
        assert acc > 0.50, \
            f"Expected accuracy > 50 % on separable data, got {acc * 100:.1f}%"


# ══════════════════════════════════════════════════════════════════════════════
# Mini training-loop (epoch + batching pattern)
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainingLoop:
    def test_mini_epoch_loop_runs_without_error(self):
        """
        Simulate 3 epochs of training on 200 synthetic samples with a
        batch size of 50.  Each epoch shuffles, splits into batches, and
        runs train_op.  No assertion beyond 'no exception thrown'.
        """
        np.random.seed(4)
        N_TOTAL = 200
        BATCH   = 50
        C, D    = 10, 784

        X_np = np.random.randn(D, N_TOTAL).astype(np.float32)
        Y_np = make_one_hot(np.arange(N_TOTAL) % C, C)

        X = vf.placeholder([D, 1])
        Y = vf.placeholder([C, 1])
        l1 = vf.Dense(D, 32, activation=vf.nn.relu)
        l2 = vf.Dense(32, C)
        logits   = l2(l1(X))
        loss     = vf.losses.softmax_cross_entropy(logits, Y)
        train_op = vf.train.AdamOptimizer(0.001).minimize(loss)

        init = vf.global_variables_initializer()
        sess = vf.Session()
        sess.run(init)

        for _epoch in range(3):
            perm   = np.random.permutation(N_TOTAL)
            Xs, Ys = X_np[:, perm], Y_np[:, perm]
            for start in range(0, N_TOTAL, BATCH):
                end  = min(start + BATCH, N_TOTAL)
                sess.run(train_op,
                         feed_dict={X: vf.Tensor(Xs[:, start:end]),
                                    Y: vf.Tensor(Ys[:, start:end])})

    def test_evaluation_after_training(self):
        """
        After 3 epochs on separable synthetic data, running logits
        outside of train_op must give consistent (non-random) predictions.
        """
        np.random.seed(5)
        C, N = 10, 200
        labels = np.arange(N) % C
        X_np   = np.random.randn(C, N).astype(np.float32) * 0.3
        for j, cls in enumerate(labels):
            X_np[cls, j] = 5.0
        Y_np = make_one_hot(labels, C)

        X = vf.placeholder([C, 1])
        Y = vf.placeholder([C, 1])
        l1 = vf.Dense(C, 32, activation=vf.nn.relu)
        l2 = vf.Dense(32,  C)
        logits   = l2(l1(X))
        loss     = vf.losses.softmax_cross_entropy(logits, Y)
        train_op = vf.train.AdamOptimizer(0.02).minimize(loss)

        init = vf.global_variables_initializer()
        sess = vf.Session()
        sess.run(init)

        BATCH = 50
        for _epoch in range(5):
            perm   = np.random.permutation(N)
            Xs, Ys = X_np[:, perm], Y_np[:, perm]
            for start in range(0, N, BATCH):
                end = min(start + BATCH, N)
                sess.run(train_op,
                         feed_dict={X: vf.Tensor(Xs[:, start:end]),
                                    Y: vf.Tensor(Ys[:, start:end])})

        # Two independent evaluation runs must agree
        out1 = sess.run(logits, feed_dict={X: vf.Tensor(X_np)}).to_numpy()
        out2 = sess.run(logits, feed_dict={X: vf.Tensor(X_np)}).to_numpy()
        np.testing.assert_array_equal(out1, out2,
            err_msg="Repeated evaluation runs gave different logits")

        # Must be better than random chance
        preds = np.argmax(out1, axis=0)
        acc   = float(np.mean(preds == labels))
        assert acc > 0.50, f"Post-training acc {acc*100:.1f}% is not above 50%"

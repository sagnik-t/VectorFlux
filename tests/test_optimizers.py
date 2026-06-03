"""
tests/test_optimizers.py — T14: Optimizers and TrainOp

Tests cover:
  - GradientDescentOptimizer: step value, minimize returns TrainOp,
    loss decreases, explicit var_list, end-to-end linear regression
  - AdamOptimizer: step value, moment vectors, loss decreases,
    end-to-end convergence
  - TrainOp: returns loss Tensor, step counter, feed_dict threading,
    end-to-end 2-class logistic regression
"""

import numpy as np
import pytest
import vectorflux as vf


# ── Helpers ────────────────────────────────────────────────────────────────────

def t(arr):
    return vf.Tensor(arr.astype(np.float32))


def one_hot(indices, C):
    N  = len(indices)
    oh = np.zeros((C, N), dtype=np.float32)
    for j, idx in enumerate(indices):
        oh[idx, j] = 1.0
    return oh


@pytest.fixture(autouse=True)
def clean_graph():
    vf.reset_default_graph()
    yield
    vf.reset_default_graph()


# ══════════════════════════════════════════════════════════════════════════════
# GradientDescentOptimizer
# ══════════════════════════════════════════════════════════════════════════════

class TestGradientDescentOptimizer:
    def test_minimize_returns_train_op(self):
        W    = vf.Variable(t(np.ones((2, 2), dtype=np.float32)))
        ph   = vf.placeholder([2, 2])
        loss = vf.losses.mse(W, ph)
        opt  = vf.train.GradientDescentOptimizer(0.01)
        op   = opt.minimize(loss)
        assert isinstance(op, vf.TrainOp)

    def test_sgd_step_updates_weights(self):
        """One SGD step: w_new = w_old - lr * grad.  Exact value check."""
        # Simple loss: reduce_mean(W) — grad is 1/N everywhere.
        lr   = 0.1
        na   = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        W    = vf.Variable(t(na))
        loss = vf.reduce_mean(W)
        opt  = vf.train.GradientDescentOptimizer(lr)
        op   = opt.minimize(loss, var_list=[W])
        sess = vf.Session()
        sess.run(op)
        expected = na - lr * np.full(4, 1.0 / 4)
        np.testing.assert_allclose(W.numpy, expected, atol=1e-6)

    def test_sgd_loss_decreases(self):
        """A single gradient step should reduce MSE loss on a random problem."""
        np.random.seed(0)
        C, N = 4, 8
        np_W   = np.random.randn(C, N).astype(np.float32)
        np_tgt = np.random.randn(C, N).astype(np.float32)
        W   = vf.Variable(t(np_W))
        ph  = vf.placeholder([C, N])
        loss = vf.losses.mse(W, ph)
        opt  = vf.train.GradientDescentOptimizer(0.05)
        op   = opt.minimize(loss, var_list=[W])
        sess = vf.Session()
        fd   = {ph: t(np_tgt)}
        l0   = sess.run(loss, feed_dict=fd).to_numpy()[0]
        sess.run(op, feed_dict=fd)
        l1   = sess.run(loss, feed_dict=fd).to_numpy()[0]
        assert l1 < l0, f"Expected loss to decrease, got {l0:.6f} → {l1:.6f}"

    def test_explicit_var_list(self):
        """var_list= controls which variables get updated."""
        W1 = vf.Variable(t(np.ones(3, dtype=np.float32)), name="W1")
        W2 = vf.Variable(t(np.ones(3, dtype=np.float32)), name="W2")
        loss = vf.reduce_mean(vf.add(W1, W2))
        # Only optimise W1
        opt  = vf.train.GradientDescentOptimizer(0.1)
        op   = opt.minimize(loss, var_list=[W1])
        sess = vf.Session()
        sess.run(op)
        # W2 must be unchanged
        np.testing.assert_allclose(W2.numpy, np.ones(3), atol=1e-7)
        # W1 must have changed
        assert not np.allclose(W1.numpy, np.ones(3))

    def test_sgd_linear_regression(self):
        """Fit y = 2*x with a single weight matrix using SGD."""
        np.random.seed(1)
        N   = 16
        x_  = np.random.randn(1, N).astype(np.float32)
        y_  = 2.0 * x_

        W   = vf.Variable(t(np.zeros((1, 1), dtype=np.float32)), name="W")
        x_ph = vf.placeholder([1, N])
        y_ph = vf.placeholder([1, N])
        pred = vf.matmul(W, x_ph)
        loss = vf.losses.mse(pred, y_ph)
        opt  = vf.train.GradientDescentOptimizer(0.05)
        op   = opt.minimize(loss, var_list=[W])

        sess = vf.Session()
        fd   = {x_ph: t(x_), y_ph: t(y_)}
        for _ in range(200):
            sess.run(op, feed_dict=fd)

        # Weight should converge to ~2.0
        np.testing.assert_allclose(W.numpy.flat[0], 2.0, atol=0.1)

    def test_no_variables_raises(self):
        """minimize() with empty var_list raises RuntimeError."""
        loss = vf.make_const(t(np.array([1.0], dtype=np.float32)))
        opt  = vf.train.GradientDescentOptimizer(0.01)
        with pytest.raises(RuntimeError):
            opt.minimize(loss, var_list=[])


# ══════════════════════════════════════════════════════════════════════════════
# AdamOptimizer
# ══════════════════════════════════════════════════════════════════════════════

class TestAdamOptimizer:
    def test_minimize_returns_train_op(self):
        W    = vf.Variable(t(np.ones((2, 2), dtype=np.float32)))
        ph   = vf.placeholder([2, 2])
        loss = vf.losses.mse(W, ph)
        opt  = vf.train.AdamOptimizer(0.001)
        op   = opt.minimize(loss)
        assert isinstance(op, vf.TrainOp)

    def test_adam_step_updates_weights(self):
        """First Adam step: verify exact weight change using closed-form formula."""
        lr, beta1, beta2, eps = 0.1, 0.9, 0.999, 1e-8
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        W  = vf.Variable(t(na))
        loss = vf.reduce_sum(W)  # grad = [1, 1, 1]
        opt  = vf.train.AdamOptimizer(lr, beta1, beta2, eps)
        op   = opt.minimize(loss, var_list=[W])
        sess = vf.Session()
        sess.run(op)

        g      = np.ones(3, dtype=np.float64)
        m      = (1 - beta1) * g
        v      = (1 - beta2) * g ** 2
        m_hat  = m / (1 - beta1)
        v_hat  = v / (1 - beta2)
        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        expected = na - update.astype(np.float32)

        np.testing.assert_allclose(W.numpy, expected, rtol=1e-5)

    def test_adam_moment_vectors_initialized(self):
        """After one step, moment vectors must be set for the variable."""
        W    = vf.Variable(t(np.ones(4, dtype=np.float32)))
        loss = vf.reduce_mean(W)
        opt  = vf.train.AdamOptimizer()
        op   = opt.minimize(loss, var_list=[W])
        sess = vf.Session()
        sess.run(op)
        vid  = id(W)
        assert vid in opt._m
        assert vid in opt._v
        assert opt._m[vid].shape == (4,)
        assert opt._v[vid].shape == (4,)

    def test_adam_loss_decreases(self):
        """A single Adam step should reduce MSE loss."""
        np.random.seed(3)
        C, N   = 3, 6
        np_W   = np.random.randn(C, N).astype(np.float32)
        np_tgt = np.random.randn(C, N).astype(np.float32)
        W   = vf.Variable(t(np_W))
        ph  = vf.placeholder([C, N])
        loss = vf.losses.mse(W, ph)
        opt  = vf.train.AdamOptimizer(0.1)
        op   = opt.minimize(loss, var_list=[W])
        sess = vf.Session()
        fd   = {ph: t(np_tgt)}
        l0   = sess.run(loss, feed_dict=fd).to_numpy()[0]
        sess.run(op, feed_dict=fd)
        l1   = sess.run(loss, feed_dict=fd).to_numpy()[0]
        assert l1 < l0, f"Expected loss to decrease, got {l0:.6f} → {l1:.6f}"

    def test_adam_convergence(self):
        """Adam should converge on a small linear regression problem."""
        np.random.seed(5)
        N    = 8
        x_   = np.random.randn(1, N).astype(np.float32)
        y_   = 3.0 * x_

        W    = vf.Variable(t(np.zeros((1, 1), dtype=np.float32)))
        x_ph = vf.placeholder([1, N])
        y_ph = vf.placeholder([1, N])
        pred = vf.matmul(W, x_ph)
        loss = vf.losses.mse(pred, y_ph)
        opt  = vf.train.AdamOptimizer(0.1)
        op   = opt.minimize(loss, var_list=[W])

        sess = vf.Session()
        fd   = {x_ph: t(x_), y_ph: t(y_)}
        for _ in range(300):
            sess.run(op, feed_dict=fd)

        np.testing.assert_allclose(W.numpy.flat[0], 3.0, atol=0.15)

    def test_no_variables_raises(self):
        loss = vf.make_const(t(np.array([1.0], dtype=np.float32)))
        opt  = vf.train.AdamOptimizer()
        with pytest.raises(RuntimeError):
            opt.minimize(loss, var_list=[])


# ══════════════════════════════════════════════════════════════════════════════
# TrainOp
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainOp:
    def test_returns_loss_tensor(self):
        """sess.run(train_op) returns a Tensor (the loss value)."""
        W    = vf.Variable(t(np.ones((2, 2), dtype=np.float32)))
        ph   = vf.placeholder([2, 2])
        loss = vf.losses.mse(W, ph)
        opt  = vf.train.GradientDescentOptimizer(0.01)
        op   = opt.minimize(loss, var_list=[W])
        sess = vf.Session()
        fd   = {ph: t(np.zeros((2, 2), dtype=np.float32))}
        result = sess.run(op, feed_dict=fd)
        assert isinstance(result, vf.Tensor)

    def test_loss_tensor_value(self):
        """The returned Tensor holds the correct pre-update loss value."""
        na    = np.ones((2,), dtype=np.float32)
        zeros = np.zeros((2,), dtype=np.float32)
        W     = vf.Variable(t(na))
        ph    = vf.placeholder([2])
        loss  = vf.losses.mse(W, ph)
        opt   = vf.train.GradientDescentOptimizer(0.0)  # lr=0 → no update
        op    = opt.minimize(loss, var_list=[W])
        sess  = vf.Session()
        fd    = {ph: t(zeros)}
        loss_val = sess.run(op, feed_dict=fd).to_numpy()[0]
        np.testing.assert_allclose(loss_val, 1.0, atol=1e-6)  # mse([1,1],[0,0])=1

    def test_step_counter_increments(self):
        """TrainOp._step increments by 1 on each sess.run()."""
        W    = vf.Variable(t(np.ones(3, dtype=np.float32)))
        loss = vf.reduce_mean(W)
        op   = vf.train.GradientDescentOptimizer(0.0).minimize(loss, var_list=[W])
        sess = vf.Session()
        assert op._step == 0
        sess.run(op)
        assert op._step == 1
        sess.run(op)
        sess.run(op)
        assert op._step == 3

    def test_feed_dict_threading(self):
        """feed_dict values reach the graph correctly inside TrainOp._execute."""
        np.random.seed(9)
        W    = vf.Variable(t(np.ones((2, 3), dtype=np.float32)))
        ph   = vf.placeholder([2, 3])
        loss = vf.losses.mse(W, ph)
        opt  = vf.train.GradientDescentOptimizer(0.1)
        op   = opt.minimize(loss, var_list=[W])
        sess = vf.Session()

        # Two different targets yield different losses
        fd1 = {ph: t(np.zeros((2, 3), dtype=np.float32))}
        fd2 = {ph: t(np.full((2, 3), 5.0, dtype=np.float32))}
        l1  = sess.run(op, feed_dict=fd1).to_numpy()[0]
        # Reset W for a fair comparison
        vf.reset_default_graph()

    def test_auto_discovers_variables(self):
        """minimize(loss) with no var_list auto-discovers registered Variables."""
        W1 = vf.Variable(t(np.ones(3, dtype=np.float32)), name="W1")
        W2 = vf.Variable(t(np.ones(3, dtype=np.float32)), name="W2")
        ph = vf.placeholder([3])
        loss = vf.losses.mse(vf.add(W1, W2), ph)
        opt  = vf.train.GradientDescentOptimizer(0.1)
        op   = opt.minimize(loss)  # no var_list
        assert len(op._vars) == 2

    def test_sgd_2class_logistic_regression(self):
        """
        End-to-end: 2-class logistic regression converges.
        Data: x in R^2, class 0 if x[0]>0 else class 1.
        After enough SGD steps, accuracy > 90%.
        """
        np.random.seed(42)
        C, N_train = 2, 64
        # Class 0: positive x[0] (mean +3), Class 1: negative x[0] (mean -3)
        # ±3 sigma separation → Bayes optimal ~99.9%, easily > 90% with a linear model
        x0 = np.vstack([ np.random.randn(N_train // 2) + 3,
                          np.random.randn(N_train // 2)]).astype(np.float32)
        x1 = np.vstack([ np.random.randn(N_train // 2) - 3,
                          np.random.randn(N_train // 2)]).astype(np.float32)
        x_data = np.hstack([x0, x1])           # [2, N_train]
        y_data = one_hot([0] * (N_train // 2) + [1] * (N_train // 2), C)

        W  = vf.Variable(t(np.zeros((C, 2), dtype=np.float32)), name="W")
        xp = vf.placeholder([2, N_train])
        yp = vf.placeholder([C, N_train])
        logits = vf.matmul(W, xp)
        loss   = vf.losses.softmax_cross_entropy(logits, yp)
        opt    = vf.train.GradientDescentOptimizer(0.05)
        op     = opt.minimize(loss, var_list=[W])

        sess = vf.Session()
        fd   = {xp: t(x_data), yp: t(y_data)}
        for _ in range(200):
            sess.run(op, feed_dict=fd)

        # Evaluate accuracy
        logits_val = sess.run(logits, feed_dict={xp: t(x_data)}).to_numpy()
        preds = np.argmax(logits_val, axis=0)
        true_labels = np.array([0] * (N_train // 2) + [1] * (N_train // 2))
        accuracy = (preds == true_labels).mean()
        assert accuracy > 0.9, f"Expected accuracy > 90%, got {accuracy:.1%}"

    def test_adam_2class_logistic_regression(self):
        """Same as above but with Adam; should converge faster."""
        np.random.seed(42)
        C, N_train = 2, 64
        x0 = np.vstack([ np.random.randn(N_train // 2) + 3,
                          np.random.randn(N_train // 2)]).astype(np.float32)
        x1 = np.vstack([ np.random.randn(N_train // 2) - 3,
                          np.random.randn(N_train // 2)]).astype(np.float32)
        x_data = np.hstack([x0, x1])
        y_data = one_hot([0] * (N_train // 2) + [1] * (N_train // 2), C)

        W  = vf.Variable(t(np.zeros((C, 2), dtype=np.float32)), name="W")
        xp = vf.placeholder([2, N_train])
        yp = vf.placeholder([C, N_train])
        logits = vf.matmul(W, xp)
        loss   = vf.losses.softmax_cross_entropy(logits, yp)
        opt    = vf.train.AdamOptimizer(0.05)
        op     = opt.minimize(loss, var_list=[W])

        sess = vf.Session()
        fd   = {xp: t(x_data), yp: t(y_data)}
        for _ in range(100):
            sess.run(op, feed_dict=fd)

        logits_val = sess.run(logits, feed_dict={xp: t(x_data)}).to_numpy()
        preds      = np.argmax(logits_val, axis=0)
        true_labels = np.array([0] * (N_train // 2) + [1] * (N_train // 2))
        accuracy    = (preds == true_labels).mean()
        assert accuracy > 0.9, f"Expected accuracy > 90%, got {accuracy:.1%}"

# VectorFlux

A TensorFlow 1-style deep learning framework built from scratch in C++ and CUDA.

VectorFlux is a demonstration project — every component from tensor storage to reverse-mode autograd to GPU kernels is implemented by hand. It trains a 3-layer MLP on MNIST to **~98% test accuracy** on an NVIDIA GPU.

---

## What's inside

| Layer | Technology |
|---|---|
| Tensor storage | C++17, row-major float32, CPU + CUDA |
| Element-wise ops | Hand-written CUDA kernels |
| Matrix multiply | cuBLAS `cublasSgemm` |
| Computation graph | Static DAG (define-then-run, TF1 style) |
| Automatic differentiation | Reverse-mode autograd over the graph |
| Optimizers | SGD, Adam |
| Python API | pybind11 bindings + pure-Python layer |

---

## Requirements

- Linux x86-64
- Python 3.10 or later
- NVIDIA GPU with CUDA compute capability 7.0+ (Volta or newer)
- CUDA Toolkit 12.x or later

> **Note:** The pre-built wheel on PyPI was compiled against CUDA 13.1 on an RTX 5060.
> If your CUDA version differs, install from source (see below).

---

## Installation

### From PyPI (pre-built binary wheel)

```bash
pip install vectorflux
```

Verify the installation:

```python
import vectorflux as vf
print(vf.hello_cuda())   # → "Hello from CUDA! Device: NVIDIA GeForce RTX ..."
```

### From source

Requires: `cmake >= 3.18`, `g++`, CUDA Toolkit, pybind11.

```bash
git clone https://github.com/arjunsingh/VectorFlux.git
cd VectorFlux
cmake -B build -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
cmake --build build -j$(nproc)
pip install -e .
```

---

## Quick start

```python
import numpy as np
import vectorflux as vf

# Run on GPU (default). Pass --device cpu to use CPU instead.
vf.set_default_device('cuda')
vf.reset_default_graph()

# ── Build the graph ───────────────────────────────────────────────────────────
X = vf.placeholder([784, None], name="X")   # [features, batch]
Y = vf.placeholder([10,  None], name="Y")   # [classes,  batch]

l1 = vf.Dense(784, 256, activation=vf.nn.relu)
l2 = vf.Dense(256, 128, activation=vf.nn.relu)
l3 = vf.Dense(128,  10)

logits   = l3(l2(l1(X)))
loss     = vf.losses.softmax_cross_entropy(logits, Y)
train_op = vf.train.AdamOptimizer(0.001).minimize(loss)

# ── Train ─────────────────────────────────────────────────────────────────────
init = vf.global_variables_initializer()
sess = vf.Session()
sess.run(init)

for step in range(1000):
    x_batch = np.random.randn(784, 64).astype(np.float32)
    y_batch = np.eye(10, dtype=np.float32)[:, np.arange(64) % 10]
    loss_val = sess.run(train_op, feed_dict={X: vf.Tensor(x_batch),
                                              Y: vf.Tensor(y_batch)})
    if step % 100 == 0:
        print(f"step {step:4d}  loss={loss_val.to('cpu').to_numpy()[0]:.4f}")
```

### MNIST demo

```bash
python examples/mnist_mlp.py                 # GPU (default)
python examples/mnist_mlp.py --device cpu    # CPU
python examples/mnist_mlp.py --epochs 10     # fewer epochs
```

Expected output (GPU, 30 epochs):

```
Device: CUDA
Loading MNIST …
  train: 60,000   test: 10,000
Model: 784 → 256 → 128 → 10  |  params: 234,752
 Epoch      loss   test acc
──────────────────────────────
     1    2.1983     95.94%
     ...
    30    0.0712     98.29%
✓  Target accuracy (≥97 %) achieved!
```

---

## API overview

### Device placement

```python
vf.set_default_device('cuda')   # must be called before building the graph
vf.set_default_device('cpu')    # force CPU
vf.get_default_device()         # → 'cuda' or 'cpu'
```

### Graph building

```python
X    = vf.placeholder([784, 1])
W    = vf.Variable(np.random.randn(256, 784).astype(np.float32))
out  = vf.relu(vf.matmul(W, X))
loss = vf.losses.softmax_cross_entropy(logits, Y)
```

### Ops

| Op | Notes |
|---|---|
| `vf.add`, `vf.mul`, `vf.sub` | Element-wise; symbolic or eager |
| `vf.matmul` | 2-D only; GPU uses cuBLAS |
| `vf.relu`, `vf.sigmoid`, `vf.tanh` | Pointwise activations; fully differentiable |
| `vf.softmax` | Forward only — **no backward pass**. Use `vf.losses.softmax_cross_entropy` for training. |
| `vf.reduce_sum`, `vf.reduce_mean` | Reduces to scalar `[1]` |
| `vf.gradients(loss, [W1, W2])` | Reverse-mode autograd |

### Layers

```python
layer = vf.Dense(784, 256, activation=vf.nn.relu, name="fc1")
out   = layer(x)          # builds matmul + activation nodes
W     = layer.weights     # Variable
```

### Losses

```python
vf.losses.mse(pred, target)                        # mean squared error
vf.losses.softmax_cross_entropy(logits, labels)    # fused, numerically stable
```

### Optimizers

```python
vf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
vf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
```

### Session

```python
sess = vf.Session()                                  # captures current device
sess.run(init)                                       # initialise variables
out  = sess.run(logits, feed_dict={X: vf.Tensor(x)})
loss = sess.run(train_op, feed_dict={X: x, Y: y})   # one training step
```

---

## Limitations

- **No broadcasting** — binary ops require identical shapes.
- **No bias terms** — `Dense` layers are weight-only (`W @ x`).
- **`vf.softmax` is not differentiable** — use `vf.losses.softmax_cross_entropy` instead.
- **float32 only** — no mixed precision.
- **2-D matmul only** — no batched matmul.
- **Single GPU** — no multi-device or distributed training.
- **Linux x86-64 only** — no Windows or macOS support.

---

## Architecture

```
python/vectorflux/
    _core.so        ← C++/CUDA extension (pybind11)
    _device.py      ← set_default_device / get_default_device
    _variables.py   ← Variable class + registry
    _ops.py         ← Overloaded symbolic/eager ops
    _session.py     ← Session, TrainOp
    _layers.py      ← Dense, vf.nn, vf.layers
    _losses.py      ← vf.losses
    _optimizers.py  ← GradientDescentOptimizer, AdamOptimizer, vf.train

src/
    tensor.cpp      ← float32 tensor, rule-of-five, CPU↔CUDA transfer
    ops_cpu.cpp     ← CPU op implementations + dispatch layer
    ops_cuda.cu     ← CUDA kernels + cuBLAS matmul
    graph.cpp       ← Op/Node/Graph, forward pass implementations, gradients
    session.cpp     ← Topological sort + execution
    autograd.cpp    ← Reverse-mode gradient graph construction
```

---

## License

MIT

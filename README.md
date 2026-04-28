# 🧠 Neural Networks from Scratch

A complete, hands-on implementation of neural networks built with **NumPy only** — no PyTorch, no TensorFlow, no magic.

Every forward pass, every gradient, every optimiser update is written explicitly so you can see exactly what's happening under the hood.

> 📺 This repository was built as a hands-on companion while following the **[Neural Networks from Scratch](https://www.youtube.com/@Vizuara)** playlist by [**Vizuara**](https://www.youtube.com/@Vizuara) on YouTube. All core ideas and mathematical foundations come from that series — this repo is my personal structured implementation and extension of those concepts.

---

## 🎯 What this repo is

This is not just a copy of a tutorial. It's a structured **learning resource** that goes from a single neuron all the way to a real image classifier, organised as:

- A clean, reusable **`core/` library** you can import and build on
- **5 progressive notebooks** — each focused on one concept, with math explanations and visualisations
- A **final project** on Fashion-MNIST using Keras, with a direct mapping back to the hand-built code

---

## 🗂️ Repository structure

```
neural-network-from-scratch/
│
├── core/                          # Reusable NumPy-only NN library
│   ├── __init__.py
│   ├── layers.py                  # DenseLayer, DropoutLayer
│   ├── activations.py             # ReLU, Softmax
│   ├── losses.py                  # CategoricalCrossEntropy, SoftmaxWithCrossEntropy
│   └── optimizers.py              # SGD, Adagrad, RMSProp, Adam
│
├── notebooks/
│   ├── 01_backprop_foundations.ipynb     # Chain rule, single neuron, one layer
│   ├── 02_core_architecture.ipynb        # OOP design: forward/backward API
│   ├── 03_optimisers_compared.ipynb      # SGD vs Adagrad vs RMSProp vs Adam
│   ├── 04_regularisation.ipynb           # L1, L2, Dropout — side-by-side
│   └── 05_fashion_mnist_project.ipynb    # Real dataset + Keras
│
└── README.md
```

---

## ⚡ Quick start

```python
from core import DenseLayer, ReLU, SoftmaxWithCrossEntropy, Adam
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
X, y = spiral_data(samples=100, classes=3)

# Build the network
dense1      = DenseLayer(2, 64)
activation1 = ReLU()
dense2      = DenseLayer(64, 3)
loss_fn     = SoftmaxWithCrossEntropy()
optimizer   = Adam(learning_rate=0.05, decay=5e-7)

# Training loop
for epoch in range(5001):
    dense1.forward(X);  activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_fn.forward(dense2.output, y)

    loss_fn.backward(loss_fn.output, y)
    dense2.backward(loss_fn.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1); optimizer.update_params(dense2)
    optimizer.post_update_params()

    if epoch % 1000 == 0:
        acc = np.mean(np.argmax(loss_fn.output, axis=1) == y)
        print(f"epoch {epoch} | loss {loss:.4f} | acc {acc:.3f}")
```

---

## 📚 Notebooks overview

### `01_backprop_foundations.ipynb`
The chain rule, explained through code. We hand-derive gradients for a single neuron, then extend to a full layer. Includes loss curves so you can see convergence happen in real time.

### `02_core_architecture.ipynb`
Introduces the object-oriented design: every component has `forward()` and `backward()` methods. Covers why we fuse Softmax + CrossEntropy (hint: the Jacobian cancels out beautifully into `ŷ - y`).

### `03_optimisers_compared.ipynb`
Trains the **exact same network** with SGD, SGD+Momentum, Adagrad, RMSProp, and Adam — then plots all curves side-by-side. The most visual notebook in the repo.

### `04_regularisation.ipynb`
Four experiments: no regularisation, L1, L2, and Dropout. Both training and test accuracy are reported so you can see when regularisation actually helps.

### `05_fashion_mnist_project.ipynb`
A real classification task: 70,000 clothing images, 10 classes. Uses Keras — but includes a direct table mapping every Keras API call back to our hand-built equivalent.

---

## 🏗️ Core library

### `core/layers.py`

| Class | Parameters | Description |
|-------|-----------|-------------|
| `DenseLayer` | `n_inputs, n_neurons, l1_reg=0, l2_reg=0` | Fully connected layer with optional L1/L2 regularisation |
| `DropoutLayer` | `drop_rate` | Inverted dropout — no rescaling needed at inference |

### `core/activations.py`

| Class | Formula | Use case |
|-------|---------|---------|
| `ReLU` | `max(0, x)` | Hidden layers |
| `Softmax` | `exp(xᵢ) / Σexp(x)` | Output layer (multiclass) |

### `core/losses.py`

| Class | Description |
|-------|-------------|
| `CategoricalCrossEntropy` | Standard CCE loss, supports sparse and one-hot labels |
| `SoftmaxWithCrossEntropy` | Fused output layer — simpler, faster backward pass |

### `core/optimizers.py`

| Class | Key hyperparameters | Description |
|-------|---------------------|-------------|
| `SGD` | `learning_rate, decay, momentum` | Vanilla + optional momentum |
| `Adagrad` | `learning_rate, decay, epsilon` | Adaptive lr via gradient accumulation |
| `RMSProp` | `learning_rate, decay, epsilon, rho` | Fixes Adagrad with EMA of grad² |
| `Adam` | `learning_rate, decay, epsilon, beta_1, beta_2` | Momentum + RMSProp + bias correction |

---

## 🧮 Concepts covered

- [x] Forward propagation
- [x] Backpropagation (chain rule)
- [x] ReLU and Softmax activations
- [x] Categorical cross-entropy loss
- [x] Fused Softmax + Loss backward pass
- [x] SGD with learning rate decay
- [x] Momentum
- [x] Adagrad
- [x] RMSProp
- [x] Adam
- [x] L1 regularisation
- [x] L2 regularisation (weight decay)
- [x] Dropout (inverted)
- [x] Applied project: Fashion-MNIST with Keras

---

## 🛠️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/neural-network-from-scratch
cd neural-network-from-scratch
pip install numpy nnfs matplotlib tensorflow
```

Then open any notebook with JupyterLab or VS Code.

---

## 🙏 Credits & References

This work stands on the shoulders of great educators. Full credit goes to:

**Primary source — the series this repo follows**
- 📺 [**Neural Networks from Scratch**](https://www.youtube.com/@Vizuara) — [Vizuara](https://www.youtube.com/@Vizuara) on YouTube
  All mathematical foundations, architectural choices, and training concepts in this repo originate from this playlist. If you find this repo useful, go watch the series — it is excellent.

**Further reading**
- [*Deep Learning*](https://www.deeplearningbook.org/) — Goodfellow, Bengio & Courville (free online)
- [cs231n Lecture Notes](https://cs231n.github.io/) — Stanford's notes on backprop and optimisers
- [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/) — Michael Nielsen (free online)

---

## 📝 License

MIT — use freely, credit appreciated.

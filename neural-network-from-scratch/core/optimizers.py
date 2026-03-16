import numpy as np


class _BaseOptimizer:
    """Shared learning-rate decay logic for all optimisers."""

    def __init__(self, learning_rate, decay):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.iterations            = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def post_update_params(self):
        self.iterations += 1


class SGD(_BaseOptimizer):
    """
    Stochastic Gradient Descent with optional momentum and learning-rate decay.

    Update rule (no momentum):
        w ← w - lr * ∇w

    Update rule (with momentum β):
        v ← β·v - lr·∇w
        w ← w + v

    Momentum lets the update "remember" past directions, which helps escape
    shallow local minima and speeds up convergence.

    Parameters
    ----------
    learning_rate : initial step size            (default 1.0)
    decay         : lr decay per step            (default 0)
    momentum      : momentum coefficient β ∈ [0, 1)  (default 0 = vanilla SGD)
    """

    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        super().__init__(learning_rate, decay)
        self.momentum = momentum

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, "weight_momentum"):
                layer.weight_momentum = np.zeros_like(layer.weights)
                layer.bias_momentum   = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentum \
                           - self.current_learning_rate * layer.dweights
            bias_updates   = self.momentum * layer.bias_momentum \
                           - self.current_learning_rate * layer.dbiases

            layer.weight_momentum = weight_updates
            layer.bias_momentum   = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates   = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases  += bias_updates


class Adagrad(_BaseOptimizer):
    """
    Adaptive Gradient algorithm.

    Accumulates the squared gradients and divides the learning rate by their
    square root, giving frequently-updated parameters a smaller effective lr.

    Update rule:
        G ← G + (∇w)²
        w ← w - lr / (√G + ε) · ∇w

    Limitation: G only grows → lr shrinks to ~0 over time.

    Parameters
    ----------
    learning_rate : initial step size   (default 1.0)
    decay         : lr decay per step   (default 0)
    epsilon       : numerical stability (default 1e-7)
    """

    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache   = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache   += layer.dbiases  ** 2

        layer.weights += -self.current_learning_rate \
            * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases  += -self.current_learning_rate \
            * layer.dbiases  / (np.sqrt(layer.bias_cache)   + self.epsilon)


class RMSProp(_BaseOptimizer):
    """
    Root Mean Square Propagation.

    Fixes Adagrad's "lr → 0" problem by using an *exponential moving average*
    of squared gradients instead of a cumulative sum.

    Update rule:
        G ← ρ·G + (1-ρ)·(∇w)²
        w ← w - lr / (√G + ε) · ∇w

    Parameters
    ----------
    learning_rate : initial step size        (default 0.001)
    decay         : lr decay per step        (default 0)
    epsilon       : numerical stability      (default 1e-7)
    rho           : smoothing factor ρ ∈ (0,1)  (default 0.9)
    """

    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon
        self.rho     = rho

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache   = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache \
                           + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache   = self.rho * layer.bias_cache \
                           + (1 - self.rho) * layer.dbiases  ** 2

        layer.weights += -self.current_learning_rate \
            * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases  += -self.current_learning_rate \
            * layer.dbiases  / (np.sqrt(layer.bias_cache)   + self.epsilon)


class Adam(_BaseOptimizer):
    """
    Adaptive Moment Estimation — the most widely used optimiser in practice.

    Combines momentum (1st moment) with RMSProp (2nd moment), and corrects for
    the initialisation bias in the first few steps.

    Update rule:
        m ← β₁·m + (1-β₁)·∇w          # 1st moment (momentum)
        v ← β₂·v + (1-β₂)·(∇w)²       # 2nd moment (RMSProp)
        m̂ = m / (1 - β₁ᵗ)             # bias-corrected
        v̂ = v / (1 - β₂ᵗ)
        w ← w - lr · m̂ / (√v̂ + ε)

    Parameters
    ----------
    learning_rate : initial step size   (default 0.001)
    decay         : lr decay per step   (default 0)
    epsilon       : numerical stability (default 1e-7)
    beta_1        : 1st-moment decay    (default 0.9)
    beta_2        : 2nd-moment decay    (default 0.999)
    """

    def __init__(self, learning_rate=0.001, decay=0.0,
                 epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon
        self.beta_1  = beta_1
        self.beta_2  = beta_2

    def update_params(self, layer):
        if not hasattr(layer, "weight_momentum"):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.weight_cache    = np.zeros_like(layer.weights)
            layer.bias_momentum   = np.zeros_like(layer.biases)
            layer.bias_cache      = np.zeros_like(layer.biases)

        # 1st moment update
        layer.weight_momentum = self.beta_1 * layer.weight_momentum \
                              + (1 - self.beta_1) * layer.dweights
        layer.bias_momentum   = self.beta_1 * layer.bias_momentum \
                              + (1 - self.beta_1) * layer.dbiases

        # 2nd moment update
        layer.weight_cache = self.beta_2 * layer.weight_cache \
                           + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache   = self.beta_2 * layer.bias_cache \
                           + (1 - self.beta_2) * layer.dbiases  ** 2

        # Bias correction
        t = self.iterations + 1
        weight_m_hat = layer.weight_momentum / (1 - self.beta_1 ** t)
        bias_m_hat   = layer.bias_momentum   / (1 - self.beta_1 ** t)
        weight_v_hat = layer.weight_cache    / (1 - self.beta_2 ** t)
        bias_v_hat   = layer.bias_cache      / (1 - self.beta_2 ** t)

        # Parameter update
        layer.weights += -self.current_learning_rate \
            * weight_m_hat / (np.sqrt(weight_v_hat) + self.epsilon)
        layer.biases  += -self.current_learning_rate \
            * bias_m_hat   / (np.sqrt(bias_v_hat)   + self.epsilon)

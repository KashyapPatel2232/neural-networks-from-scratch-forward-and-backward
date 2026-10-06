"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
import numpy as np
def numerical_gradient(f, x, eps=1e-5):
    # TODO: Estimate the gradient of scalar f w.r.t. array x via central finite differences
    grad = np.zeros_like(x)
    if x.size == 0:
        return x

    for index in np.ndindex(x.shape):
        dum = x[index]
        x[index] = x[index] + eps
        f_plus = f(x)
        x[index] = dum
        x[index] = x[index] - eps
        f_minus = f(x)
        x[index] = dum
        grad[index] = (f_plus - f_minus) / (2*eps)

    return grad

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    analytic_grad = np.asarray(analytic_grad, dtype=float)
    numeric_grad = np.asarray(numeric_grad, dtype=float)

    denominator = np.maximum(
        np.maximum(np.abs(analytic_grad), np.abs(numeric_grad)),
        tol
    )
    relative_errors = np.abs(analytic_grad - numeric_grad) / denominator
    return float(np.max(relative_errors))

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
  """Create a fully connected layer.

  Inputs:
    in_dim: int, input feature size
    out_dim: int, output feature size
    weight_init_fn: callable(in_dim, out_dim) -> (W, b)

  Returns layer dict with keys:
    params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
    forward(x) -> (y, cache) with y shape (batch, out_dim)
    backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
      Analytic dx/dW/db must match numerical_gradient via gradient_check.
  """
  # TODO: your approach here
  W, b = weight_init_fn(in_dim, out_dim)
  def forward(x):
    out = x @ W + b
    return out, x       # cache is the input
  def backward(dout, x):
    dx = dout @ W.T
    dW = x.T @ dout
    db = np.sum(dout, axis = 0)
    grads = {'W': dW, 'b': db}
    return dx, grads
  
  params = {'W': W, 'b' : b}
  
  return {'forward':forward, 'backward':backward, 'params':params}

def init_fn(in_dim, out_dim):
  rng = np.random.RandomState(0)
  W = rng.randn(in_dim, out_dim) * 0.1
  b = np.zeros(out_dim)
  return W, b

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    # TODO: your approach here
    if kind == 'relu':
      def forward(x):
        out = np.maximum(0, x)
        return out, x
      
      def backward(dout, x):
        dx = dout * (x>0)
        return dx, {}

    params = {}
    
    return {'forward':forward, 'backward':backward, 'params':params}



layer = make_activation('relu')
x = np.array([[-1.0, 0.5, 2.0]])          # shape (batch=1, dim=3)
y, cache = layer['forward'](x)             # y.shape == (1, 3)
dx, param_grads = layer['backward'](np.ones_like(y), cache)
# dx.shape == (1, 3); param_grads == {}
print(y.shape, dx.shape, param_grads, layer['params'])

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement


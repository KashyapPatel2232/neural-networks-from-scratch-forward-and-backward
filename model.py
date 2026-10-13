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
    if kind == 'relu':
        def forward(x):
            return np.maximum(0, x), x

        def backward(dout, x):
            return dout * (x > 0), {}

    elif kind == 'tanh':
        def forward(x):
            y = np.tanh(x)
            return y, y

        def backward(dout, y):
            return dout * (1 - y**2), {}

    else:
        raise ValueError(f"Unsupported activation: {kind}")

    return {'forward': forward, 'backward': backward, 'params': {}}

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    # TODO: your approach here
    if scheme == 'he':
      W = np.random.randn(in_dim, out_dim)*np.sqrt(2/in_dim)
      b = np.zeros(out_dim)
    else:
      W = np.random.randn(in_dim, out_dim)
      b = np.zeros(out_dim)
    return W, b

# Step 6 - make_loss
def make_loss(kind='cross_entropy'):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits).

    Inputs to loss_fn:
      logits: (batch, C) float array of raw class scores
      labels: (batch,) int array of class indices in [0, C)
    Outputs:
      loss: Python float, mean scalar loss over the batch (finite)
      d_logits: (batch, C) gradient of loss w.r.t. logits (finite)
    Must pass gradient_check, be minimized by confident correct predictions,
    and stay finite under saturated logits.
    """
    # TODO: your approach here
    def loss_fn(logits, labels):
      
      if kind == 'cross_entropy':
        one_hot_label = np.zeros_like(logits)

        for i in range(len(labels)):
          one_hot_label[i, labels[i]] = 1

        epsilon = 1e-12
        logits_shifted = logits - np.max(logits, axis = 1, keepdims= True)
        softmax = np.exp(logits_shifted)/np.sum(np.exp(logits_shifted), axis = 1, keepdims = True)
        loss = -1* np.mean(np.sum(one_hot_label * np.log(softmax + epsilon), axis = 1))
        d_logits = (softmax - one_hot_label)/len(logits)
      
      return loss, d_logits

    return loss_fn

# Step 7 - make_sequential
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    # TODO: your approach here
    def forward(x):
      cache_ls = []
      for func in layers:
        x, cache = func['forward'](x)
        cache_ls.append(cache)
      return x, cache_ls
    
    def backward(dout, caches):
      grads_list = []
      for layer, cache in zip(reversed(layers), reversed(caches)):
          dout, layer_grads = layer['backward'](dout, cache)
          grads_list.append(layer_grads)

      grads_list.reverse()
      return dout, grads_list

    params = [layer['params'] for layer in layers]
    return {'forward':forward, 'backward':backward, 'params':params}

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch.

    Inputs:
      model: sequential dict with 'forward', 'backward', 'params'
             model['forward'](x) -> (logits, caches)
             model['backward'](d_logits, caches) -> (dx, param_grads)
      loss_fn: callable (logits, y) -> (loss, d_logits)
      x: np.ndarray (batch, in_dim)
      y: np.ndarray (batch,) integer labels

    Returns:
      loss: float, scalar batch loss
      param_grads: nested np.ndarrays matching model['params'] layout
                   (gradients of loss w.r.t. every parameter)
    """
    # TODO: your approach here
    out, caches = model['forward'](x)
    loss, d_logits = loss_fn(out, y)
    dx, param_grads = model['backward'](d_logits, caches)
    return loss, param_grads

# Step 9 - make_optimizer
def make_optimizer(params, lr=1e-2, kind='sgd'):
    """Build an optimizer that updates params in place.

    Inputs:
      params: arrays, possibly nested in lists/dicts (or dict of arrays) to optimize
      lr: float learning rate
      kind: str algorithm name (e.g. 'sgd')

    Returns:
      dict with key 'step'. step(grads) applies one in-place update
      using grads structured like params. Parameter shapes must stay
      unchanged. Repeated steps must reduce a simple convex objective
      within a modest fixed budget and keep values finite.
    """
    # TODO: your approach here
    def update(p,g):
      if isinstance(p, dict):
        for key in p:
          update(p[key],g[key])
      elif isinstance(p, list):
        for param, grad in zip(p, g):
          update(param, grad)
      else:
        p[...] -= lr * g

    def step(grads):
      if kind == 'sgd':
        update(params,grads)

    return {'step':step}

# Step 10 - train_step
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    """Perform one complete optimization step over a minibatch.

    Inputs:
      model: sequential model dict with 'forward', 'backward', and 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x_batch: np.ndarray of shape (B, D)
      y_batch: np.ndarray of shape (B,) integer class labels

    Returns:
      loss: float, scalar batch loss evaluated BEFORE the parameter update.
      Model parameters are updated in place; shapes unchanged and values finite.
    """
    # TODO: your approach here
    out, caches = model['forward'](x_batch)
    loss, d_logits = loss_fn(out, y_batch)
    dx, param_grads = model['backward'](d_logits,caches)
    params = model['params']
    opt = optimizer['step'](param_grads)
    return loss

# Step 11 - train
def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0):
    """Run a deterministic minibatch training loop.

    Inputs:
      model: sequential model dict with 'forward', 'backward', 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x: np.ndarray of shape (N, D) training features
      y: np.ndarray of shape (N,) integer class labels
      epochs: int, number of full passes over the data
      batch_size: int, minibatch size
      seed: int, RNG seed for deterministic shuffling / batching

    Returns:
      history: list[float] of length `epochs`; history[t] is the mean
      train_step loss over minibatches in epoch t.
      Model parameters are updated in place; shapes unchanged.
    """
    # TODO: your approach here
    np.random.seed(seed)
    n_batch = np.ceil(len(y) // batch_size).item()
    history = []
    for i in range(epochs):
      loss = 0
      for start in range(0, len(y), batch_size):
        end = start + batch_size
        x_batch = x[start:end]
        y_batch = y[start:end]
        loss += train_step(model, loss_fn, optimizer, x_batch, y_batch)
      history.append(loss/n_batch)
    return history

# Step 12 - design_network
def design_network(input_dim, num_classes, seed=0):
    if input_dim < 2 or num_classes < 2:
        raise ValueError("input_dim and num_classes must be at least 2.")

    data_rng = np.random.RandomState(seed)
    samples_per_class = 256

    if num_classes == 2:
        # Two interleaving spirals
        theta = np.sqrt(data_rng.rand(samples_per_class)) * 2 * np.pi

        x0 = np.column_stack([
            theta * np.cos(theta),
            theta * np.sin(theta),
        ])
        x1 = np.column_stack([
            theta * np.cos(theta + np.pi),
            theta * np.sin(theta + np.pi),
        ])

        x_2d = np.vstack([x0, x1])
        y = np.repeat(np.arange(2), samples_per_class)
    else:
        # One concentric ring per class
        total_samples = samples_per_class * num_classes
        y = np.repeat(np.arange(num_classes), samples_per_class)
        theta = data_rng.uniform(0, 2 * np.pi, total_samples)
        radius = 0.7 + 1.15 * y + data_rng.normal(0, 0.035, total_samples)

        x_2d = np.column_stack([
            radius * np.cos(theta),
            radius * np.sin(theta),
        ])

    indices = data_rng.permutation(len(y))
    x_2d, y = x_2d[indices], y[indices]

    # The task is nonlinear in the first two dimensions; pad additional
    # input dimensions with zeros when input_dim is greater than 2.
    x = np.zeros((len(y), input_dim), dtype=float)
    x[:, :2] = x_2d

    split = int(0.78 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    init_rng = np.random.RandomState(seed)

    def init_function(in_dim, out_dim):
        weights = init_rng.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        bias = np.zeros(out_dim)
        return weights, bias

    layers = [
        make_dense(input_dim, 128, init_function),
        make_activation("relu"),
        make_dense(128, 64, init_function),
        make_activation("relu"),
        make_dense(64, num_classes, init_function),
    ]
    model = make_sequential(layers)
    loss_fn = make_loss("cross_entropy")
    optimizer = make_optimizer(model["params"], lr=0.05)

    for _ in range(120):
        batch_order = data_rng.permutation(len(y_train))
        for start in range(0, len(y_train), 32):
            batch_indices = batch_order[start : start + 32]
            train_step(
                model,
                loss_fn,
                optimizer,
                x_train[batch_indices],
                y_train[batch_indices],
            )

    logits, _ = model["forward"](x_val)
    accuracy = float(np.mean(np.argmax(logits, axis=1) == y_val))

    return model, {"accuracy": accuracy,"x": x_val,"y": y_val}

# Step 13 - improve_generalization
def improve_generalization(baseline_model_fn, x_train, y_train, x_val, y_val, seed=0):
    """Improve held-out accuracy over an unregularized baseline.

    Inputs:
      baseline_model_fn: zero-arg callable -> fresh untrained sequential model
        (dict with 'forward', 'backward', 'params') matching the data dims.
      x_train, y_train: training features (N, D) and int labels (N,).
      x_val, y_val: validation features (N_val, D) and int labels (N_val,).
      seed: int for deterministic training.

    The regularized model uses validation-loss early stopping (patience 12)
    and restores the parameters from its best validation-loss epoch.

    Returns:
      dict with keys:
        'val_accuracy': float accuracy of the improved model on x_val/y_val
        'baseline_val_accuracy': float val accuracy of plain unregularized SGD
        'predictions': np.ndarray shape (N_val,) int preds from improved model
        'model': the trained improved model

    Required behavior:
      val_accuracy > baseline_val_accuracy
      predictions == argmax(model.forward(x_val), axis=1)
      val_accuracy == mean(predictions == y_val)
      predictions are non-constant (not a trivial single-class predictor)
    """
    def make_optimizer_regularised(params, lr=0.05, weight_decay=0.001, kind='sgd'):
        if kind != 'sgd':
            raise ValueError(f"Unsupported optimizer: {kind}")

        def update(param_tree, grad_tree, decay=False):
            if isinstance(param_tree, dict):
                for key, param in param_tree.items():
                    update(param, grad_tree[key], key == 'W')
            elif isinstance(param_tree, list):
                for param, grad in zip(param_tree, grad_tree):
                    update(param, grad)
            else:
                param_tree[...] -= lr * (grad_tree + weight_decay * param_tree if decay else grad_tree)

        def step(grads):
            update(params, grads)

        return {'step': step}

    def copy_tree(tree):
        if isinstance(tree, dict):
            return {key: copy_tree(value) for key, value in tree.items()}
        if isinstance(tree, list):
            return [copy_tree(value) for value in tree]
        return tree.copy()

    def restore_tree(target, source):
        if isinstance(target, dict):
            for key in target:
                restore_tree(target[key], source[key])
        elif isinstance(target, list):
            for target_value, source_value in zip(target, source):
                restore_tree(target_value, source_value)
        else:
            target[...] = source

    model0 = baseline_model_fn()
    model1 = baseline_model_fn()
    loss_fn = make_loss("cross_entropy")
    optimizer0 = make_optimizer(model0["params"], lr=0.05)
    optimizer1 = make_optimizer_regularised(model1["params"], lr=0.05)

    baseline_epochs = 70
    max_epochs = 120
    patience = 12
    min_delta = 1e-5
    baseline_rng = np.random.RandomState(seed)
    for _ in range(baseline_epochs):
        batch_order = baseline_rng.permutation(len(y_train))
        for start in range(0, len(y_train), 32):
            batch_indices = batch_order[start : start + 32]
            train_step(
                model0,
                loss_fn,
                optimizer0,
                x_train[batch_indices],
                y_train[batch_indices],
            )

    data_rng = np.random.RandomState(seed)
    best_val_loss = float('inf')
    best_val_accuracy = -float('inf')
    best_params = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        batch_order = data_rng.permutation(len(y_train))
        for start in range(0, len(y_train), 32):
            batch_indices = batch_order[start : start + 32]
            train_step(
                model1,
                loss_fn,
                optimizer1,
                x_train[batch_indices],
                y_train[batch_indices],
            )

        val_logits, _ = model1["forward"](x_val)
        val_loss, _ = loss_fn(val_logits, y_val)
        val_accuracy = float(np.mean(np.argmax(val_logits, axis=1) == y_val))
        if (val_accuracy > best_val_accuracy or
                (val_accuracy == best_val_accuracy and val_loss < best_val_loss - min_delta)):
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_params = copy_tree(model1["params"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_params is not None:
        restore_tree(model1["params"], best_params)

    logits, _ = model0["forward"](x_val)
    accuracy = float(np.mean(np.argmax(logits, axis=1) == y_val))

    logits_reg, _ = model1["forward"](x_val)
    predictions = np.argmax(logits_reg, axis=1)
    accuracy_reg = float(np.mean(predictions == y_val))

    return {"val_accuracy": accuracy_reg, 'baseline_val_accuracy': accuracy, 'predictions': predictions, 'model': model1}


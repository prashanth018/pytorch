import torch
from torch.nn import (
    Linear,
    Parameter,
    Sequential,
    ReLU,
    Dropout,
    BatchNorm1d,
    Module,
    BCEWithLogitsLoss,
    Conv2d,
    MaxPool2d,
    Flatten,
    CrossEntropyLoss,
    functional,
)
from torch.optim import Adam
from torch.optim.optimizer import Optimizer


def make_tensor():
    """Return a 2x3 float32 tensor [[1, 2, 3], [4, 5, 6]]."""
    return torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)


def reshape_transpose(t):
    """Reshape a 1D tensor of 6 elements to 2x3 (row-major) and return its transpose.

    Args:
        t (torch.Tensor): 1D tensor with exactly 6 elements.

    Returns:
        torch.Tensor: Transpose of the 2x3 reshape, with shape (3, 2).
    """
    assert t.shape == (6,)
    return t.reshape(2, 3).T


def grad_of_square(x_val):
    """Return dy/dx for y = x**2 at x = x_val using autograd.

    Args:
        x_val (float): scalar input value.

    Returns:
        float: gradient of x**2 w.r.t. x at x_val.
    """
    x = torch.tensor(x_val, requires_grad=True)
    y = x**2
    y.backward()
    return x.grad.item()


def grad_wss(w_list, x_list):
    """Build w (requires_grad) and x from lists, compute
    loss = 0.5 * sum((w * x)**2), backward, return w.grad
    as a list of floats rounded to 4 decimals.
    """
    assert len(w_list) == len(x_list)
    w = torch.tensor(w_list, dtype=torch.float32, requires_grad=True)
    # print(w)
    x = torch.tensor(x_list, dtype=torch.float32)
    loss = 0.5 * ((w * x) ** 2).sum()
    # print(loss)
    loss.backward()
    return torch.round(w.grad, decimals=4).tolist()
    # return [round(w_i.item(), 4) for w_i in list(w.grad)]


def gen_inp_for_linear_reg():
    X = ((torch.rand(18).reshape(6, 3) * 2) - 1) * 6
    w = torch.tensor([2.0, -4.0, 3.0])
    y = torch.round(X @ w)
    return (X, y)


def gen_inp_2_layer_mlp():
    x = torch.tensor([[1.0, 2.0]])
    w1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b1 = torch.tensor([0.0, 0.0])
    w2 = torch.tensor([[1.0, 1.0]])
    b2 = torch.tensor([0.0])
    return (x, w1, b1, w2, b2)


# Final w = tensor([ 1.9724, -4.0179,  3.0264]), b = tensor([-0.1865])
# Expected w = tensor([2.0, -4.0, 3.0]), b = tensor([0.0])
def fit_linear_regression(X, y, lr=0.1, steps=500):
    """Fit y ~= X @ w + b with full-batch GD using only autograd.

    Args:
        X: Float tensor (N, D)
        y: Float tensor (N,) or (N, 1)
        lr: learning rate
        steps: number of gradient descent iterations

    Returns:
        w: Float tensor (D,) learned weights (no grad)
        b: Float tensor scalar learned bias (no grad)
    """
    if len(y.shape) > 1:
        y = y.squeeze(1)
    assert len(X.shape) == 2
    batch, dim = X.shape
    w = torch.rand(dim, dtype=torch.float32, requires_grad=True)
    b = torch.rand(1, dtype=torch.float32, requires_grad=True)
    print("init w = ", w, "init b = ", b)
    for st in range(steps):
        # print(
        #     "\n\n#####\nep = ",
        #     st,
        # )
        y_pred = X @ w + b
        # print("y_pred", y_pred)
        # print("y_    ", y)
        loss = (y - y_pred) ** 2
        print("E = ", st, "loss = ", loss.mean().item())
        loss.mean().backward()
        with torch.no_grad():
            # print("w = ", w, "b = ", b)
            # print("w.grad = ", w.grad, "b.grad = ", b.grad)
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad = None
        b.grad = None

    # y_pred = X @ w + b
    # loss = (y - y_pred) ** 2
    # loss.mean().backward()
    # print("loss ", loss)
    return w.detach(), b.detach()


def single_neuron_forward(x):
    """Forward pass of one fixed linear neuron.

    Args:
        x: torch.Tensor of shape (1, 3).

    Returns:
        Python float, the neuron output.
    """
    neuron = Linear(3, 1)
    with torch.no_grad():
        neuron.weight = Parameter(torch.tensor([[0.5, -0.2, 0.3]]))
        neuron.bias = Parameter(torch.tensor([0.1]))
    return neuron(x).item()


def two_layer_mlp_forward(x, w1, b1, w2, b2):
    """Build a 2-layer MLP, set fixed weights, return scalar output.

    Args:
        x (torch.Tensor): Input of shape (1, 2).
        w1 (torch.Tensor): First Linear weight, shape (2, 2).
        b1 (torch.Tensor): First Linear bias, shape (2,).
        w2 (torch.Tensor): Second Linear weight, shape (1, 2).
        b2 (torch.Tensor): Second Linear bias, shape (1,).

    Returns:
        float: Scalar network output.
    """
    n1 = Linear(2, 2)
    with torch.no_grad():
        n1.weight = Parameter(w1)
        n1.bias = Parameter(b1)
    n2 = Linear(2, 1)
    with torch.no_grad():
        n2.weight = Parameter(w2)
        n2.bias = Parameter(b2)
    net = Sequential(n1, ReLU(), n2)
    return net(x).item()


def relu(t):
    """Element-wise ReLU: max(0, t).

    Args:
        t (torch.Tensor): input tensor

    Returns:
        torch.Tensor: activated tensor
    """
    return torch.where(t > 0.0, t, 0.0)


def leaky_relu(t, slope=0.01):
    """Element-wise Leaky ReLU with given negative slope.

    Args:
        t (torch.Tensor): input tensor
        slope (float): slope for negative values

    Returns:
        torch.Tensor: activated tensor
    """
    return torch.where(t > 0.0, t, slope * t)


def softmax(t, dim):
    """Numerically stable softmax along dim.

    Args:
        t (torch.Tensor): input tensor
        dim (int): dimension along which to apply softmax

    Returns:
        torch.Tensor: tensor of same shape as t; slices along dim sum to 1
    """
    exp = torch.exp(t - t.max(dim=dim, keepdim=True).values)
    return exp / exp.sum(dim=dim, keepdim=True)


def mse(pred, target):
    """
    Compute mean squared error between pred and target.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground-truth values (same shape as pred).

    Returns:
        float: Mean of squared differences.
    """
    assert pred.shape == target.shape
    return torch.mean((pred - target) ** 2).item()


def bce_with_logits(logits, targets):
    """Mean BCE-with-logits loss, numerically stable, rounded to 4 decimals.

    Args:
        logits (torch.Tensor): 1-D raw logits.
        targets (torch.Tensor): 1-D binary targets in {0, 1}, same shape.

    Returns:
        float: mean loss rounded to 4 decimal places.
    """
    # Formula = max(x,0) - x*y + log(1 + e^-|x|)
    return (
        (
            torch.clamp(logits, min=0.0)
            - logits * targets
            + torch.log(1.0 + torch.exp(-torch.absolute(logits)))
        )
        .mean()
        .item()
    )


def dropout_demo():
    """Demonstrate Dropout behavior in eval vs train mode.

    Returns:
        tuple: (eval_output, train_nonzero_count)
            eval_output: result of Dropout(ones) in eval mode (identity)
            train_nonzero_count: int count of nonzero elements after Dropout in train mode
    """

    torch.manual_seed(0)
    x = torch.ones(10)
    drop = Dropout(p=0.5)
    drop.eval()
    eval_out = drop(x)
    drop.train()
    train_out = drop(x)
    # print(train_out)
    count = torch.count_nonzero(train_out).item()
    return (eval_out, count)


def bn_eval(x, mean, var, gamma, beta, eps=1e-5):
    """Apply batch-norm inference normalization.

    Args:
        x (Tensor): input tensor
        mean (Tensor): running mean
        var (Tensor): running variance
        gamma (Tensor): scale parameter
        beta (Tensor): shift parameter
        eps (float): numerical stability constant

    Returns:
        Tensor: normalized and affine-transformed tensor
    """
    return torch.round(gamma * ((x - mean) / ((var + eps) ** 0.5)) + beta, decimals=4)


class RegularizedMLP(Module):
    """MLP with BatchNorm1d and Dropout for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_p: float = 0.3):
        super().__init__()
        self.net = Sequential(
            Linear(in_features=input_dim, out_features=hidden_dim),
            BatchNorm1d(num_features=hidden_dim),
            ReLU(),
            Dropout(p=dropout_p),
            Linear(in_features=hidden_dim, out_features=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return shape (N,) logits for batch x of shape (N, input_dim)."""
        out = self.net(x)
        # print(out.shape)
        return out.squeeze(1)


def train_mlp_model(model, X_train, y_train, epochs=150, lr=1e-2):
    """Train model in-place with BCEWithLogitsLoss + Adam. Return model."""
    model.train()
    optim = Adam(model.parameters(), lr=lr)
    bce_loss_fn = BCEWithLogitsLoss()
    # print("y_pred at the start", torch.sigmoid(model(X_train)))
    for e in range(epochs):
        model.zero_grad()
        y_pred = model(X_train)
        # print("X_train", X_train)
        # print("y_train", y_train)
        # print("y_pred", y_pred)
        loss = bce_loss_fn(input=y_pred, target=y_train)
        loss.backward()
        optim.step()
        # print("epoch = ", e, "loss = ", loss.item())
    # y_pred = model(X_train)
    return model


def generate_bce_train_data(batch_size=32, input_dim=10):
    return (
        torch.rand((batch_size, input_dim)),
        torch.randint(low=0, high=2, size=(batch_size,)) * 1.0,
    )


def run_regularized_mlp():
    input_dim = 12
    batch_size = 80
    model = RegularizedMLP(input_dim=input_dim)
    x, y = generate_bce_train_data(batch_size=batch_size, input_dim=input_dim)
    model = train_mlp_model(model, x, y, epochs=100)
    y_pred = model(x)
    print(y_pred)
    print("y_pred logits", y_pred)
    print("y_pred at the end", torch.sigmoid(y_pred))
    print("y_train", y)


def conv_out_shape(h, w, kernel, stride, padding):
    """Return (H_out, W_out) for a 2D conv with the given spatial params.

    Args:
        h: input height
        w: input width
        kernel: kernel size (same for H and W)
        stride: stride (same for H and W)
        padding: padding (same for H and W)

    Returns:
        Tuple of ints (H_out, W_out).
    """

    def foo(val):
        return (val + 2 * padding - kernel) // stride + 1

    return (foo(h), foo(w))


def apply_conv2d():
    """Build nn.Conv2d(1, 1, kernel_size=2, bias=False), set a fixed kernel, convolve a fixed input.

    Under torch.no_grad(), set weight to tensor([[[[1.0, 0.0], [0.0, 1.0]]]]).
    Input is tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]).

    Returns:
        torch.Tensor: Output of shape (1, 1, 2, 2).
    """
    conv = Conv2d(in_channels=1, out_channels=1, kernel_size=2, bias=False)
    with torch.no_grad():
        conv.weight = Parameter(torch.eye(2).unsqueeze(0).unsqueeze(0))
    input = torch.arange(
        start=1, end=10, dtype=torch.float32, requires_grad=False
    ).reshape(
        (1, 1, 3, 3),
    )
    print(input)
    return conv(input).detach()


def generate_conv_train_data(batch_size=100, channels=1, image_size=(8, 8)):
    labels = torch.randint(low=0, high=2, size=(batch_size,))
    return (
        torch.rand((batch_size, channels, *image_size)),
        functional.one_hot(labels).float(),
    )


class TinyCNN(Module):
    """Small CNN: Conv2d -> ReLU -> pool -> (optional extras) -> Linear."""

    def __init__(self, img_size=8, n_classes=2):
        super().__init__()
        self.net = Sequential(
            # in_conv1     = (n, 1, 8, 8)
            Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1),
            # out_conv1    = (n, 6, 8, 8)   [kernel=3, padding=1, stride=1] -> (8+2-3)/1+1 = 8
            ReLU(),
            # out_relu1    = (n, 6, 8, 8)   [elementwise, shape unchanged]
            MaxPool2d(3, stride=1),
            # out_pool1    = (n, 6, 6, 6)   [kernel=3, padding=0, stride=1] -> (8+0-3)/1+1 = 6
            Conv2d(in_channels=6, out_channels=3, kernel_size=3, padding=1),
            # out_conv2    = (n, 3, 6, 6)   [kernel=3, padding=1, stride=1] -> (6+2-3)/1+1 = 6
            ReLU(),
            # out_relu2    = (n, 3, 6, 6)   [elementwise, shape unchanged]
            MaxPool2d(3, stride=1),
            # out_pool2    = (n, 3, 4, 4)   [kernel=3, padding=0, stride=1] -> (6+0-3)/1+1 = 4
            Flatten(),
            # out_flatten  = (n, 48)        [3*4*4 = 48, batch dim kept]
            Linear(in_features=48, out_features=n_classes),
            # out_linear   = (n, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def build_model(img_size=8, n_classes=2):
    """Return an instance of your TinyCNN (or equivalent nn.Module)."""
    return TinyCNN(img_size=img_size, n_classes=n_classes)


def train_conv_model(model, train_x, train_y, optim, epochs=15, batch_size=32, seed=0):
    """Train model on train_x/train_y and return the trained model.

    Args:
        model: nn.Module from build_model
        train_x: FloatTensor (N, 1, H, W)
        train_y: LongTensor (N,)
        epochs: number of full passes over the data
        lr: optimizer learning rate
        batch_size: mini-batch size
        seed: RNG seed for shuffling / init determinism

    Returns:
        Trained model (same instance is fine).
    """
    torch.manual_seed(seed=seed)
    ce_loss_fn = CrossEntropyLoss()
    model.train()
    # print("####Start")
    # indices = torch.randperm(train_x.shape[0])[:4]
    # pred_y = model(train_x[indices])
    # print("pred_y", pred_y)
    # print("pred_y softmax", torch.round(torch.softmax(pred_y, dim=1), decimals=4))
    # print("\n")
    for e in range(epochs):
        # sample minibatch of indices
        indices = torch.randperm(train_x.shape[0])[:batch_size]
        model.zero_grad()
        pred_y = model(train_x[indices])
        loss = ce_loss_fn(input=pred_y, target=train_y[indices])
        loss.backward()
        optim.step()
        print("Epoch e = ", e, "Loss = ", loss.item())

    return model


def test_conv_nets():
    x_train, y_train = generate_conv_train_data(
        batch_size=96, channels=1, image_size=(8, 8)
    )
    # print("x_train ", x_train.shape)
    # print("y_train ", y_train.shape)
    model = build_model(img_size=8, n_classes=2)
    model = train_conv_model(
        model,
        x_train,
        y_train,
        optim=Adam(params=model.parameters(), lr=0.01),
        epochs=500,
        batch_size=8,
        seed=0,
    )
    indices = torch.randperm(x_train.shape[0])[:4]
    pred_y = model(x_train[indices])
    print("train_y", y_train[indices])
    print("pred_y logits", pred_y)
    print("pred_y softmax", torch.round(torch.softmax(pred_y, dim=1), decimals=4))


def sgd_step(w, grad, lr):
    """Perform one SGD update step.

    Args:
        w: Current parameter tensor.
        grad: Gradient tensor (same shape as w).
        lr: Learning rate (float).

    Returns:
        Updated parameter tensor w - lr * grad.
    """
    return w - lr * grad


def momentum_step(w, grad, v, lr, mu):
    """One SGD-with-momentum step.

    Args:
        w: parameter tensor
        grad: gradient tensor (same shape as w)
        v: velocity tensor (same shape as w)
        lr: learning rate (float)
        mu: momentum coefficient (float)

    Returns:
        (w_new, v_new) tuple of tensors
    """
    # momentum is kinda the running mean of past grads
    v_new = mu * v + grad
    # use this new grad to compute weight
    w_new = w - lr * v_new
    return (w_new, v_new)


def adam_step(w, grad, m, v, t, lr, beta1, beta2, eps):
    """One Adam update with bias-corrected moments.

    Args:
        w: current parameters (torch.Tensor)
        grad: gradient of the loss w.r.t. w (torch.Tensor)
        m: first moment estimate (torch.Tensor)
        v: second moment estimate (torch.Tensor)
        t: timestep, 1-indexed (int)
        lr: learning rate (float)
        beta1: exp. decay for first moment (float)
        beta2: exp. decay for second moment (float)
        eps: numerical stability constant (float)

    Returns:
        Tuple (w_new, m_new, v_new) as torch.Tensor values.
    """
    m_new = (beta1 * m) + (1 - beta1) * grad
    v_new = (beta2 * v) + (1 - beta2) * grad**2

    # t is 1 indexed
    m_hat = m_new / (1 - beta1**t)
    v_hat = v_new / (1 - beta2**t)

    w_new = w - lr * m_hat / (v_hat**0.5 + eps)
    return (w_new, m_new, v_new)


class MyOptimizer(Optimizer):
    """
    Design your own optimizer!
    - You can base it on SGD, RMSProp, Adam, or create something new.
    - Must subclass torch.optim.Optimizer.
    - Only dense gradients are supported.
    """

    def __init__(self, params, lr=1e-3):
        # You can add your own hyperparameters here
        defaults = dict(lr=lr)
        self.eps = 1e-8
        self.b1 = 0.9
        self.b2 = 0.999
        super().__init__(params, defaults)

    @torch.no_grad()
    def adam_step(self, param, grad, _lr):
        # self.state[param]["m"] = (self.b1 * self.state[param]["m"]) + (1 - self.b1) * grad
        self.state[param]["m"].mul_(self.b1).add_(grad, alpha=1 - self.b1)
        # self.state[param]["v"] = (self.b2 * self.state[param]["v"]) + (1 - self.b2) * grad**2
        self.state[param]["v"].mul_(self.b2).addcmul_(grad, grad, value=1 - self.b2)

        # bias correction
        bc1 = 1 - self.b1 ** self.state[param]["t"]
        bc2 = 1 - self.b2 ** self.state[param]["t"]
        # m_hat = self.state[param]["m"] / (bc1)
        # v_hat = self.state[param]["v"] / (bc2)
        # param = param - lr * m_hat / (v_hat**0.5 + self.eps)
        param.addcdiv_(
            self.state[param]["m"],
            self.state[param]["v"].sqrt().div_(bc2**0.5).add_(self.eps),
            value=-_lr / bc1,
        )
        self.state[param]["t"] += 1

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            # for example group["params"] == model.parameters()
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p not in self.state:
                    self.state[p]["m"] = torch.zeros_like(p)
                    self.state[p]["v"] = torch.zeros_like(p)
                    self.state[p]["t"] = 1

                grad = p.grad
                self.adam_step(param=p, grad=grad, _lr=lr)

        return loss


def test_model_params():
    model = build_model(img_size=8, n_classes=2)
    for name, p in model.named_parameters():
        print("name = ", name, " params = ", p.shape)


def test_optimizer():
    x_train, y_train = generate_conv_train_data(
        batch_size=96, channels=1, image_size=(8, 8)
    )
    # print(x_train.shape)
    # print(y_train.shape)
    model = build_model(img_size=8, n_classes=2)
    model = train_conv_model(
        model=model,
        train_x=x_train,
        train_y=y_train,
        optim=MyOptimizer(params=model.parameters(), lr=0.01),
        epochs=1,
        batch_size=8,
        seed=0,
    )
    indices = torch.randperm(x_train.shape[0])[:4]
    y_pred = model(x_train[indices])
    print("y_train ", y_train[indices])
    print("y_pred ", y_pred)
    print("softmax(y_pred) ", torch.round(torch.softmax(y_pred, dim=1), decimals=4))


if __name__ == "__main__":
    # print(make_tensor())
    # print(reshape_transpose(torch.arange(1, 7, dtype=torch.int32)))
    # print(grad_of_square(3.0))
    # print(grad_wss([1, 2, 3], [4, 5, 6]))
    # print(grad_wss([1.0, 2.0], [3.0, 4.0]))
    # X, y = gen_inp_for_linear_reg()
    # print("X = ", X, "y = ", y)
    # print(fit_linear_regression(X, y, lr=0.01, steps=500))
    # print(single_neuron_forward(torch.tensor([[1.0, 2.0, 3.0]])))
    # two_layer_mlp_forward()
    # print(two_layer_mlp_forward(*gen_inp_2_layer_mlp()))
    # print(relu(torch.tensor([-2.0, -0.5, 0.0, 1.5])))
    # print(leaky_relu(torch.tensor([-2.0, -0.5, 0.0, 1.5]), 0.1))
    # print(softmax(torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), dim=1))
    # print(mse(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])))
    # print(bce_with_logits(torch.tensor([0.0, 2.0]), torch.tensor([0.0, 1.0])))
    # print(dropout_demo())
    # print(
    #     bn_eval(
    #         torch.tensor([1.0, 2.0, 3.0]),
    #         torch.tensor(2.0),
    #         torch.tensor(1.0),
    #         torch.tensor(1.0),
    #         torch.tensor(0.0),
    #     )
    # )
    # run_regularized_mlp()
    # print(conv_out_shape(28, 28, 5, 2, 0))
    # print(apply_conv2d())
    # test_conv_nets()
    # print(sgd_step(torch.tensor([1.0, 2.0]), torch.tensor([0.5, 1.0]), lr=0.1))
    # print(
    #     momentum_step(
    #         torch.tensor([1.0, 2.0]),
    #         torch.tensor([0.1, 0.2]),
    #         torch.tensor([0.0, 0.0]),
    #         0.1,
    #         0.9,
    #     )
    # )
    # print(
    #     adam_step(
    #         torch.tensor([1.0, 2.0]),
    #         torch.tensor([0.5, -0.5]),
    #         torch.zeros(2),
    #         torch.zeros(2),
    #         1,
    #         0.001,
    #         0.9,
    #         0.999,
    #         1e-8,
    #     )
    # )
    test_optimizer()
    # test_conv_nets()
    pass

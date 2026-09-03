import torch
from torch.nn import Linear, Parameter, Sequential, ReLU, Dropout


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
    print(dropout_demo())
    # pass

import torch


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
    for st in range(steps):
        y_pred = X @ w + b
        loss = (y - y_pred) ** 2
        loss.mean().backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad = None
        b.grad = None

    y_pred = X @ w + b
    loss = (y - y_pred) ** 2
    loss.mean().backward()
    print(loss)

    # TODO: return detached w, b
    return 1.0


if __name__ == "__main__":
    # print(make_tensor())
    # print(reshape_transpose(torch.arange(1, 7, dtype=torch.int32)))
    # print(grad_of_square(3.0))
    # print(grad_wss([1, 2, 3], [4, 5, 6]))
    # print(grad_wss([1.0, 2.0], [3.0, 4.0]))
    pass

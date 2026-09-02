# pytorch

Practice repo. Keeping PyTorch fluency sharp — tensors, shapes, broadcasting, autograd.

## What's here

- `main.py` — my own PyTorch puzzles. Small self-contained functions with a docstring spec: build a tensor, reshape/transpose, take a gradient, etc.
- `puzzles.py` — Sasha Rush's Tensor Puzzles — https://github.com/srush/Tensor-Puzzles
  - 21 puzzles, each solved with broadcasting only. No for-loops, no `view`, no comparison operators outside the allowed set.
  - The point is to stop thinking in loops and start thinking in shapes.

## Notes

- Env: `ml-env`

## To-do
- [ ] test `fit_linear_regression`

## Learning
- .backward() only works on scalars. if the output is a vector you either reduce it first (e.sum().backward()) or tell it what gradient is coming in (e.backward(torch.ones_like(e))).
- the graph is built during the forward pass and freed during backward. so if you call backward twice on the same forward it errors out. you have to run the forward again to get a fresh graph.
- `u.grad = None` only clears the gradient for that vector.
- Run `zero_grad` logic to zero the accumulated gradients.
- Re: "In-place modification of a leaf tensor that requires grad is blocked." error observed for `w -= lr * w.grad`, autograd won't let you update the params youself after running the backward. This is because it holds a reference to all the params and if you update any of these params, the values become inconsistent with the computed grads. For you to want to update any params, you'd have to put the updates under `with torch.no_grad()` block.
- Given above, you should also not do `w = w - lr * w.grad`. This block builds a new object of `w` (this one doesn't have `w.grad`) and builds a computational graph between the new w to the old w. The output wouldn't be wrong but a new copy of w gets created each epoch. After 100 steps you've got a 100-link chain hanging off w, and none of it can be freed. So, update inplace under `with torch.no_grad()` block.
- Ran a simple linear regression experiment. Final weights are supposed to be `[2.0, -4.0, 3.0]`. w tensor started as `[0.0954, 0.2741, 0.2490]`, w.grad was `[14.9288,  90.1841, -99.0611]` after the first epoch. This essentially conveys, w[0] should go down a bit, w[1] should go down a lot, w[2] should go up a lot. Directions were almost as it should be. 
- Initialized lr to `0.1` accidentally and this blew up the loss eventually reaching nan in 5 epochs.
- `detach` helps you detach 1 edge from the graph and return the tensor without any gradient associated, this is so that the callers will not accidentally back prop the gradient of the object being returned. `no_grad` for places where you don't want to construct any computational graph and grads for the parameters. Where exactly is `no_grad` used? Say, you have a training loop of 100 epochs after which, you want to inference and see how the agent behaves. You enclose the inference inside `no_grad` so that the gradients don't pool. You also use `no_grad` for updating the params yourself.
- `torch.inference_mode()` is a stricter `no_grad` — it also skips version tracking, so it's a bit faster.
- `Parameter` is a subclass of Tensor. `Parameter` toggles `requires_grad=True` by default. Its a special object controlled by `Module`. 
- `Module` tracks all the parameters. Module is responsible for moving the parameters to optimizer, to the GPU and to save.
- Neuron weight matrix shape is always of shape `(out_dim, in_dim)`. We do `X @ w.T + b`.
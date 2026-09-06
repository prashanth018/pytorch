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
- [x] test `fit_linear_regression`

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
- `keepdim=True` preserves the dimensions of the source matrix. This helps in operations like norm and softmax.
- `dropout.eval()` puts module in eval mode. `dropout.train()` puts module in train mode.
- `sum(dim=1)`, `max(dim=1)`, `softmax(dim=1)` all of these mean 'do the operation along the dim=1 (col)' -> sum all the cols, max along the col, softmax along the col etc.,
- [DEBUGGING] I swapped the entries to loss function `bce_loss_fn(input=y_train, target=y_pred)`. Softmax must have gotten applied to the y_train. y_pred stayed as logits. Loss ended being meaningless and continued to be -ve. 
- In simple case, Conv layer translates `(1, in_channel, H, W)` to `(1, out_channel, out(H), out(W))`. `in_channel` and `out_channel` are our preferences. Kind of like how we specify `in` & `out` dims for `Linear` layer. We don't have to specify `H`, `W`, `out(H)` & `out(W)`. Model figures it out itself. 
- Why does H & W have to vary? because of the kernel size, padding and stride and other values provided into the conv net. `out(H) = ((H + 2*padding - kernel) // stride) + 1`
- What makes `out_channels` vary? Model dynamically stacks additional `kernels` to increase the number of out channels. For example, if its a Conv layer of `Conv2d(3, 16, kernel_size=3)`. Weight matrix will be of shape `(16, 3, 3, 3)` -> This means, 16 kernels of size `(3, 3, 3)` convolved over `(3, H, W)` to generate `(3, out(H), out(W))` each. Therefore, output dim is `(16, 3, out(H), out(W))`.
- `AdaptiveAvgPool2d`: Once you are done with all the Convs, you'd have to have blast these features off to a dense linear layer before inferencing logits. How does the SOTA architectures figure the input dims for Linear layer? `Linear(in_dims=??, out_dims=1)`. `MaxPool2d` takes kernel size and gives you an ouput spacial size `(H,W)` based on the kernel size. `AdaptiveAvgPool2d` takes `required output` as an input and figures the kernel and stride to match the required output. Example: `AdaptiveAvgPool2d((1,1))`.
- Max Pooling conversion formula, say input has height `H` then `out(H) = ((H + 2*padding - kernel) // stride) + 1`.
- **Max Pooling `stride = kernel` by default.
- `k[torch.arange(len(e)), e] = 1.0` and `k.scatter_(1, e[:, None], 1.0)` and `F.one_hot(e, 2).float()` are all equally efficient way to generate 1 hot prob vector from label vector `e`.


### Losses
- Sigmoid is different from Softmax.
- Sigmoid = `1/(1 + e^-x)` is used when the output is when its a multi-label classification. Softmax = `e^x1/sum(e^x)` is used when the output is a prob distribution
- `BCEWithLogits` uses sigmoid underneath. `CrossEntropy` uses softmax underneath. BCEWithLogits with C=2 forces only 1 class to be `1.0` (behaves like softmax). BCEWithLogits with C>2 *doesn't* force mutual exclusion. This is because it doesn't treat logits as a probability vector. Instead it just forces the logit between 0 and 1. Contrary, `Softmax` (CrossEntropy loss) forces the logits between 0 to 1 and normalizes them. This *forces* the mutual exclusion. This means, you can't do multi-class classification with softmax. Since `BCEWithLogits` doesn't force this, you can do multi-class classification with it. Also, you could use `BCEWithLogits` (with C>2) honor mutual exclusion, but the empirically performs worse than `CrossEntropy`. 

### Training
- An epoch is one full pass over the whole dataset. One minibatch is one step. Say `N=10000` & `batch_size=32` , `1 epoch = N / batch_size = 313 steps`.
    - Full-batch GD = All N per epoch.
    - Mini-batch GD = SGD = K mini-batches per epoch.


## Optimizers
- `w = w - lr * grad` is a simple update rule. Now the problem with this update it its too noisy. Instead we could take a running mean of the grads, like `m = beta * m + (1 - beta) * grad` where m is first moment estimate (kinda like the running mean of grads). Similarly we have `v = beta * v + (1 - beta) * grad**2` where v is the second moment estimate (like a running mean of square of grads).



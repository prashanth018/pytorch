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
- [ ] add batchnorm2d to conv net and retrain with multiple seeds, check paran norms and activations 

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
- What makes `out_channels` vary? Model dynamically stacks additional `kernels` to increase the number of out channels. For example, if its a Conv layer of `Conv2d(3, 16, kernel_size=3)`. Weight matrix will be of shape `(16, 3, 3, 3)` -> This means, 16 kernels of size `(3, 3, 3)` convolved over `(3, H, W)` to generate `(3, out(H), out(W))` each. Therefore, output dim is `(16, 3, out(H), out(W))`. `(3, 3, 3)` consolidates to `(1,3)` values. There are `16` such values stacked up vertically. 
- `AdaptiveAvgPool2d`: Once you are done with all the Convs, you'd have to have blast these features off to a dense linear layer before inferencing logits. How does the SOTA architectures figure the input dims for Linear layer? `Linear(in_dims=??, out_dims=1)`. `MaxPool2d` takes kernel size and gives you an ouput spacial size `(H,W)` based on the kernel size. `AdaptiveAvgPool2d` takes `required output` as an input and figures the kernel and stride to match the required output. Example: `AdaptiveAvgPool2d((1,1))`.
- Max Pooling conversion formula, say input has height `H` then `out(H) = ((H + 2*padding - kernel) // stride) + 1`.
- **Max Pooling `stride = kernel` by default.
- `k[torch.arange(len(e)), e] = 1.0` and `k.scatter_(1, e[:, None], 1.0)` and `F.one_hot(e, 2).float()` are all equally efficient way to generate 1 hot prob vector from label vector `e`.
- `model.named_parameters()` returns a list of layer name & set of weight & bias matrix for each layer. Example:
        ```
        name =  net.0.weight  params =  torch.Size([6, 1, 3, 3])
        name =  net.0.bias  params =  torch.Size([6])
        ```
- `torch.zeros_like(p)` instead of `torch.zeros(p.shape)`. First one matches the dtype too.
- `TensorDataset` (sub class of `Dataset`) indexes tensors along the first dimension, return sample upon query like `dataset.__getitem__(4)`.
- `DataLoader(dataset=dataset, batch_size=4, shuffle=False)` takes a dataset, preferred batch_size and shuffle settings. This class gives you an iterator of (X_train, y_train) mini_batches.
- `TinyNet` in `build_mnist_model` is the right way to build a model.
    - Mainly, you have `BatchNorm2d` that makes sure we don't run into 0ing grads like we did before.
    - out_dim consistently go up, which is what is desired in a Conv Net (aking to ResNet & AlexNet).
    - `AdaptiveMaxPool2d` makes it easier to focus on the `output(H,W)` instead of figuring out the math for kernel, padding, stride.
    - Highest %age of params in the final dense layer. 



### Losses
- Sigmoid is different from Softmax.
- Sigmoid = `1/(1 + e^-x)` is used when the output is when its a multi-label classification. Softmax = `e^x1/sum(e^x)` is used when the output is a prob distribution
- `BCEWithLogits` uses sigmoid underneath. `CrossEntropy` uses softmax underneath. BCEWithLogits with C=2 forces only 1 class to be `1.0` (behaves like softmax). BCEWithLogits with C>2 *doesn't* force mutual exclusion. This is because it doesn't treat logits as a probability vector. Instead it just forces the logit between 0 and 1. Contrary, `Softmax` (CrossEntropy loss) forces the logits between 0 to 1 and normalizes them. This *forces* the mutual exclusion. This means, you can't do multi-class classification with softmax. Since `BCEWithLogits` doesn't force this, you can do multi-class classification with it. Also, you could use `BCEWithLogits` (with C>2) honor mutual exclusion, but the empirically performs worse than `CrossEntropy`. 

### Training
- An epoch is one full pass over the whole dataset. One minibatch is one step. Say `N=10000` & `batch_size=32` , `1 epoch = N / batch_size = 313 steps`.
    - Full-batch GD = All N per epoch.
    - Mini-batch GD = SGD = K mini-batches per epoch.


### Optimizers
- `w = w - lr * grad` is a simple update rule. Now the problem with this update it its too noisy. Instead we could take a running mean of the grads, like `m = beta * m + (1 - beta) * grad` where m is first moment estimate (kinda like the running mean of grads). Similarly we have `v = beta * v + (1 - beta) * grad**2` where v is the second moment estimate (like a running mean of square of grads).
- Important note!!! timesteps corresponding to params are 1-indexed. I implemented as 0 indexed and ran into NaNs.
- use in-place ops (`mul_`, `add_`, `addcmul_`, `addcdiv_`) instead of `a*b` / `a+b` when doing tensor math in the optimizer. every out-of-place op allocates a whole new tensor. the optimizer step is pure elementwise work so it's memory-bandwidth bound, therefore those extra allocations and the reads/writes that come with them are most of the cost. (BTW, this has nothing to do with the autograd graph, the step runs under `no_grad` which means it doesn't create a computational graph. This optimization is purely to reduce the GPU compute time.)
- Refer to `MyOptimizer.adam_step` for getting a sense of how to write complex math with inplace methods. 


### Debugging Session

#### Case 1: Loss staying stagnent at 0.69 (ln2) for random seeds

This signifies that the model is performing like a random classifier (especially for binary labeled data).

Turns out, second conv layer's output (just before relu2) has all its outputs (`x*w + b`) for all the 96 images in the dataset as negative (All of them). Largest value was `-0.055`. ReLU is shorting all these params to `0`. Therefore, the flatten layer is receiving a zero matrix.

**Why is this a problem?** When we compute `loss.backward()`, given that ReLU makes the output 0.0, the incoming gradient backproped to conv2 (and thereby conv1) is 0.0. This means the `weights` and `bias` for all the layers until this ReLU stay frozen. This means all the following epochs continue to inference zero vectors from this layer.

`lr = 0.001` helped a bit by not letting the params not wander too far off.

**What can prevent this behavior?**
- **BatchNorm**: before the ReLU. It re-centres activations to roughly zero mean every batch, so a whole layer can't drift negative.
- **LeakyReLU**: Gradient is 0.01 instead of 0 on the negative side — a small leak, but enough to pass the gradient.

**Debugging:**
- Given that this exercise was about writing a new optimizer, I suspected if there was something wrong with my optimizer. I ran another experiment and swapped my optimizer with reference pytorch adam optimizer. loss values matched to 4 digits with same seed and same init data.
- Few seeds succeeded and few seeds failed.
- loss was `ln(2) = 0.693`, this means the model might be outputting something constant.
- Hypothesis: "output doesn't depend on input". So to test the hypothesis, I computed the standard deviation of the activation for the whole batch of inference. It turned out to be 0.0.
    ```python
    h = x
    for layer in model.net:
        h = layer(h)
        print(h.std(dim=0).mean().item())
    ```
- Confirmed the hypothesis by looking at the weights before the activation layer and turns out the `max` of the weights was `-0.055`.

#### First things to check when a model won't train

```python
for n, p in model.named_parameters():
    print(n, p.grad.norm().item() if p.grad is not None else "NO GRAD")
```

Zeros mean something is blocking the path. None means it's not connected to the loss at all. Huge values mean explosion. Then compute the loss, and check whether the output varies with the input.


### Rabbit hole of binary cross entropy, forward & backward KL:
*****Rabbit hole incomplete, pending thinking around this**
- What is entropy? What is "cross" in the BCE?
- What is entropy of a function? - is it same as predictability?
- Still can't explain `- P(x) * log(Q(x))` intuitively or derive it from blank paper. 
- Why is it called NLL?

- Binary cross entropy is a forward KL.

Formal cross entropy between 2 distributions P & Q. `H(P,Q) = - [P(Class 1). log(Q(Class 1)) + P(Class 0). log(Q(Class 0))]`

`Entropy of a System (P,Q) = Entropy(P) + Forward KL(P || Q)`
Entropy = Chaos = Unpredictability
When we train a NN, we try to reduce the entropy of the system. In this case, `Chaos (P, Q)`. `Tunable variables` in the system are the ones that can directly influence the outcome of System `Q`. W.r.t the tunable params, the GT is a constant. So entropy of the system = `Forward KL(P || Q)`. 


The general definition of Cross Entropy is: For every possible outcome, multiply the Reality of it happening by the Model's Surprise if it happens.
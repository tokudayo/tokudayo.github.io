---
title: "Fastest Matrix NMS in the West (Part 1): A Naive Implementation"
excerpt: "Introduction to Matrix NMS, plus a naive implementation and its problems."
header:
  teaser: "assets/images/torch.png"
tags: 
  - code
  - pytorch
  - computer-vision
  - CUDA
  - optimization
---

This is the first part of a series discussing and optimizing the Matrix NMS algorithm, which was introduced in [SOLOv2](https://arxiv.org/abs/2003.10152). The official repository is [here](https://github.com/aim-uofa/AdelaiDet/). The code for this series is available [here](https://github.com/tokudayo/fast-matrix-nms).

In the first part, I will give a brief introduction to the algorithm. We will then implement a naive version of the algorithm in PyTorch and run it on a GPU. We will see that the naive implementation is not very efficient and discuss the reasons behind it. In the next part, we will start optimizing the algorithm.

## Matrix NMS
Matrix NMS is one of the many variants of NMS (Non-Maximum Suppression) algorithms. Its goal is to remove redundant bounding boxes or masks that have high overlap with each other. It is a crucial component of many segmentation and detection pipelines, especially in the context of deep learning.

To keep the reading clear and concise, from now on we limit the topic to bounding boxes only. The same idea can be transferred to masks.

The most widely-used variant of NMS is the original NMS algorithm, or Hard NMS. Basically, it "removes boxes with high overlap with the box with the highest confidence score, then repeat". The algorithm is simple and easy to implement. The time complexity is $$O(N^2)$$, where $$N$$ is the number of boxes. The effect of removing boxes after each iteration may slightly reduce the running time. Best case is $$O(N)$$.

<!-- Given boxes coordinates $$B \in \mathcal{R}^{N\times 4}$$ where each box $$B_i$$ is represented by the top left and bottom right corners $$(x_1,y_1,x_2,y_2)$$; and their confidence scores $$s \in \mathcal{R}^N$$, the algorithm repeats the following process:

- Select the candidate box $$c \in B$$ with the highest confidence score $$max_i s_i$$  and add to the final set $$F$$.

- Remove all boxes in $$B$$ with *high overlap* with $$c$$. Typically, the algorithm iterates through all boxes in $$B$$ and removes the ones with IOU (Intersection over Union) greater than a threshold $$t$$.

- If $$B$$ is empty, stop. Otherwise, go back to step 1. The final set $$F$$ contains the selected boxes. -->



Soft NMS takes a slightly different approach: instead of removing boxes with high overlap to the selected best box, it reduces their confidence scores by a function of the IoU. In some cases it avoids removing correct predictions that just happen to have high overlap with the selected box. The time complexity is still $$O(N^2)$$ and the worst case, best case, and average case are all the same. In practice, it is usually slower than Hard NMS.

Both algorithms process the boxes in a sequential fashion in a double-nested for loop. They are not suitable for parallel computing. To remedy this, Wang et al. proposed a parallel version of NMS, called Matrix NMS [^1]. The algorithm keeps the core idea of non-max suppression, that is to suppress boxes with high overlap with the maximum/higher confidence boxes. Let's call the set of all boxes $$f$$ with confidence score higher than a box $$b$$ its *father boxes*. The algorithm decays each box score with correlation to its *probability of being suppressed by father boxes* - denoted as $$p(b)$$. Assuming each box is suppressed by at most one father box, such decay degree $$d_b$$ is the maximum decay a father box $$f_i$$ can apply to $$b$$ among all $$i$$. This in turn is affected by:

[^1]: <https://arxiv.org/abs/2003.10152>

<!-- $$d_b \underset{\sim}{\propto} p(b) \underset{\sim}{\propto} \arg \max_i  p(b, f_i)$$

where $$p(b, f_i)$$ is the probability of $$b$$ being suppressed by father box $$f_i$$. -->

- $$IoU(b, f_i)$$: theIoUbetween $$b$$ and $$f_i$$. The higher the IoU, the higher the probability of $$b$$ being suppressed by $$f_i$$.

- $$p(f_i)$$: The probability of $$f_i$$ itself being suppressed by other boxes. The higher the probability, the higher the probability of $$b$$ is being suppressed and being suppressed by $$f_i$$.

Technically, $$p(b)$$ now depends on $$p(f_i)$$, which needs to be computed prior to $$p(b)$$ and now we are stuck in the same sequential processing situation. To solve this, the authors noticed that $$p(f_i)$$ has correlation with $$\max_j IoU(f_i, g_j)$$, where $$g_j$$ are the father boxes of $$f_i$$.

$$d_b$$ can now be obtained via applying some transformation $$\mathcal{F}$$ to the two factors:

<center>
 $$d_b = \max_i \frac{\mathcal{F}(iou(b, f_i))}{\mathcal{F}(\max_j iou(f_i, g_j))}$$
 </center>

If we reformulate $$d_b$$ as the multiplicative decay factor of $$b$$'s confidence score rather can the decay degree, this becomes:

<center>
 $$d_b = \min_i \frac{\mathcal{F}(IoU(b, f_i))}{\mathcal{F}(\max_j IoU(f_i, g_j))}$$
 </center>

This enables parallel computation opportunities. Now, $$\mathcal{F}$$ should be some monotonically decreasing function. The authors choose a linear function and a gaussian function. After computing $$d_b$$ for all boxes, the algorithm then decays the confidence scores of all boxes by $$d_b$$. The final set $$F$$ contains the boxes with confidence scores higher than a threshold $$t$$.

See [the paper](https://arxiv.org/abs/2003.10152) for more details.

## Naive `torch` implementation
### Implementation
We first sort the boxes and scores in descending score order. This means box $$b_i$$ has father $$b_j~~\forall j<i$$.

For each box, we need to compute the IoU between the box and all its father boxes. This should be computed for all boxes to obtain a $$N\times N$$ IoU matrix. Here is the slightly modified self IoU matrix implementation from [`torchvision.ops.boxes`](https://pytorch.org/vision/main/_modules/torchvision/ops/boxes.html#box_iou) using existing PyTorch functions:

```python
def self_iou_matrix(boxes):
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    lt = torch.max(boxes[:, None, :2], boxes[:, :2])  # [N,M,2]
    rb = torch.min(boxes[:, None, 2:], boxes[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area[:, None] + area - inter

    return torch.triu(inter / union, dim=1)
```

The original function `box_iou` computes the IoU between two sets of boxes $$a$$ and $$b$$. For our purpose, we have modified it for the case where $$a=b$$.

We need to mask out the boxes that are not father boxes of the current box. We used `torch.triu(..., dim=1)` to obtain the upper triangular part of the matrix. This is what is used in the original implementation. The `dim=1` parameter excludes the diagonal elements.

The next step is to evaluate $$\max_j IoU(f_i, g_j)$$ for all $$f_i$$. This is just the column-wise maximum of the IoU matrix. We can use `torch.max(..., dim=0).values` to do just that.

```python
iou_matrix = self_iou_matrix(boxes)
ious_cmax = iou_matrix.max(dim=0).values # or max(dim=0)[0]
```

Now we calculate 

 $$D = \frac{\mathcal{F}(IoU(b, f_i))}{\mathcal{F}(\max_j IoU(f_i, g_j))} \in \mathcal{R}^{N \times N}$$

 to obtain the candidate decay factor matrix. The column-wise minimum of this matrix is the final decay factor $$d \in \mathcal{R}^N$$ for each box; calculate the new scores and filter out the boxes with scores below the threshold.

The complete code should look like this:

```python
def matrix_nms(boxes, scores, threshold):
    # Sort by conf
    scores, sorted_indices = scores.sort(descending=True)
    boxes = boxes[sorted_indices]

    # IoU matrix
    ious = self_iou_matrix(boxes, boxes)

    # p(f_i) estimation & enable broadcasting
    ious_cmax = ious.max(0)[0].view(-1, 1)

    # Decay factor
    if use_gaussian:
        decay = torch.exp((ious_cmax.square() - ious.square()) / gaussian_sigma)
    else:
        decay = (1 - ious) / (1 - ious_cmax)
    decay = decay.min(0)[0]

    # Select boxes
    decayed_scores = scores * decay
    indices = torch.where(decayed_scores > threshold)[0]

    return sorted_indices[indices]
```

### Performance issues
PyTorch, NumPy and similar libraries provide a high-level interface to tensor manipulation. Under the hood, they are implemented in C/C++, which is much faster than pure Python. For example, this piece of code

```python
# a: Tensor[N,]
# b: Tensor[N,]
# c: Tensor[N, N]
for i in range(N):
    for j in range(N):
        if (j > i) c[i, j] = a[i] * b[j]
```

runs much lower than this one, although on the surface it should have less ops:

```python
# a: Tensor[N,]
# b: Tensor[N,]
# c: Tensor[N, N]
c = a[:, None] @ b[None, :].triu(diagonal=1)
```

Thus, as Python devs, we are used to vectorizing our code to speed up the computation by relying on the efficient implementations of the libraries. However, in our case, this can backfire. Our Python host calls so many kernel functions that performs redundant computation. Each kernel when launched has to access the global GPU memory once for each **entry** in the input tensors which takes hundreds of cycles. The self IoU matrix is hideous to compute because of the many kernel launches, and the rest also contains redundancies.

So, we need to go lower level. To avoid launch overhead and other Python-related problem, we can use LibTorch C++. PyTorch provides a C++ frontend API with the same functions, but it is out of the question. On CPU, the runtime of Matrix NMS is much worse than Hard NMS; on GPU, we may avoid some C calls overhead and enjoy compiler's optimizations but cannot address the memory access problem and operations redundancies.

We have to go as low as rewriting the kernel functions in CUDA, which is what we will do in the next part.
---
title: "Multiprocessing in Python and PyTorch"
excerpt: "Introduction to `multiprocessing` and `torch.multiprocessing`"
header:
  teaser: "assets/images/torch.png"
tags: 
  - python
  - pytorch
---

This is the first part of a 3-part series covering multiprocessing, distributed communication, and distributed training in PyTorch.

In this post, we will cover the basics of multiprocessing in Python first, then move on to PyTorch; so even if you don't use PyTorch, you may still find helpful resources here :)

Multiprocessing is process-level parallelism, in the sense that each spawned process is allocated seperate memory and resources. In Python, in many cases, multiprocessing is used to bypass the infamous GIL, which is a global lock that prevents the interpreter from running any other code while it is executing a single thread.

## `multiprocessing`
In native Python, multiprocessing is achieved by using the `multiprocessing` module. 

The official documentation of `multiprocessing` is [here](https://docs.python.org/3/library/multiprocessing.html), and it is great! In this tutorial we will only cover 
some of the most important and relevant features of the module; for more details, please refer to the official documentation.

### Process

To spawn a new process, we create a `Process` object, and then we call the `start()` method.

```python
import multiprocessing as mp
import time

def foo(x):
    print(f"foo({x})")
    time.sleep(2)
    return x

p = mp.Process(target=foo, args=(1,))
p.start()
p.join()
```
```
foo(1)
---
Runtime: 2.0 seconds
```

Very straightforward! The `join()` method blocks the main process until the spawned process finishes. Without it, the main process would exit immediately at the end without waiting for `foo(1)` to complete.

#### Cross-process communication
For communication between two processes, we can use a `Pipe` object. For communication between two or more processes, we can use a `Queue` object.
    
```python
import multiprocessing as mp
import time

def foo(x, q):
    # Producer
    print(f"Putting {x} in queue")
    q.put(x)

def bar(q):
    # Consumer
    print(f"Got {q.get()} from queue")

queue = mp.Queue()

p1 = mp.Process(target=foo, args=(1, queue))
p2 = mp.Process(target=bar, args=(queue,))

p1.start()
p2.start()

p1.join()
p2.join()
```
```
Putting 1 in queue
Got 1 from queue
---
Runtime: 0.0 seconds
```

### Pool
`multiprocessing.Pool` creates a pool of processes, each of which is allocated a separate memory space. It is a context manager, so it can be used in a `with` statement.

```python
import multiprocessing as mp

with mp.Pool(processes=4) as pool:
    # do something
```

Otherwise, be sure to close the pool when you are done with it.

```python
pool = mp.Pool(processes=4)
# do something
pool.close()
pool.join()
```

Once `pool.close()` is invoked, no more tasks can be submitted to the pool. Once all tasks are completed, the worker processes will exit (gracefully). On the other hand, if you want to terminate the pool immediately, you can use `pool.terminate()`.

If you want to wait for all tasks to finish, you can use `pool.join()`. One must call `close()` or `terminate()` before using `join()`.

To do some actual work, in most cases you would want to use either `apply`, `map`, `starmap` or their `_async` variants.

#### `apply`
We can submit a function to the pool to be executed in a worker process by using `pool.apply`.

```python
import time
import multiprocessing as mp

def foo(x, y):
    time.sleep(3)
    return x + y

with mp.Pool(processes=4) as pool:
    a = pool.apply(foo, (1, 2))
    b = pool.apply(foo, (3, 4))
    print(a, b)
```
```
3 7
---
Runtime: 6.0 seconds
```
We create a pool with 4 worker processes, and then submit two tasks to the pool to run. Since `apply` is a blocking call, the main process will wait until the first task is completed before submitting the second task. This is mostly useless, because no parallelism is achieved here. In this case, if we want to run multiple tasks in parallel, we should use `apply_async` like this

```python
with mp.Pool(processes=4) as pool:
    handle1 = pool.apply_async(foo, (1, 2))
    handle2 = pool.apply_async(foo, (3, 4))

    a = handle1.get()
    b = handle2.get()

    print(a, b)
```
```
3 7
---
Runtime: 3.0 seconds
```
`apply_async` is non-blocking and returns a `AsyncResult` object immediately. We can then use `get` to get the result of the task.

Note that `get` will block until the task is completed; `apply(fn, args, kwargs)` is equivalent to `apply_async(fn, args, kwargs).get()`.

On the other hand, we can add a callback to `apply_async` to be executed when the task is completed:

```python
def callback(result):
    print(f"Got result: {result}")

with mp.Pool(processes=4) as pool:
    handle1 = pool.apply_async(foo, (1, 2), callback=callback)
    handle2 = pool.apply_async(foo, (3, 4), callback=callback)
```
```
Got result: 3
Got result: 7
---
Runtime: 3.0 seconds
```
In rare cases the second function may be completed before the first one, and correspondingly, the callback for the second function will be called before the callback for the first function.

Because the number of worker processes is limited, if all workers are busy when a new task is submitted, the task will be queued and executed later.
```python
with mp.Pool(processes=2) as pool:
    for _ in range(3):
        pool.apply_async(foo, (1, 2))
```
```
---
Runtime: 6.0 seconds
```
In the example above, the first and second `foo` calls are executed in the 2 workers, but the third has to wait until a worker becomes available.

#### `map` and `starmap`
`map` divides the input iterable into chunks and submits each chunk to the pool as a separate task. The results of the tasks are then gathered and returned as a list.

```python
import multiprocessing as mp
import time

def foo(x):
    print(f"Starting foo({x})")
    time.sleep(2)
    return x

with mp.Pool(processes=2) as pool:
    result = pool.map(foo, range(10), chunksize=None)
    print(result)
```
```
Starting foo(0)
Starting foo(2)
Starting foo(1)
Starting foo(3)
Starting foo(4)
Starting foo(6)
Starting foo(5)
Starting foo(7)
Starting foo(8)
Starting foo(9)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
---
Runtime: 12.0 seconds
```

In the example above, `chunksize` is set to its default value `None`. I'm not sure how the chunk size is determined, but it seems to scale with the length of the iterable argument. In this case, the chunksize is automatically calculated to be 2. This means the iterable is divided into 5 chunks of size 2: `[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]`. At first, the first two chunks are submitted to the 2 workers, and then the next two chunks are submitted. Finally, the last chunk `[8, 9]` is submitted to the either worker: at that point only one worker would process this chunk. This is why the runtime is 12 seconds, which is sub-optimal. In this case, if we explicitly set the chunksize to 1 or 5, the runtime will be 10 seconds, which is as good as it gets.

`map` is a blocking call, so it will wait until all tasks are completed before returning. Similar to `apply`, we can use `map_async` to submit tasks to the pool and get the results asynchronously.

```python
with mp.Pool(processes=2) as pool:
    handle = pool.map_async(foo, range(10), chunksize=None)
    # do something else
    result = handle.get()
    print(result)
```

The limitation of `map` is it simply passes the elements of the iterable to a function. Thus if we want to apply a multi-argument function, we either have to pass in a list and unpack it inside the function, which is ugly, or use `starmap`. For each element of the iterable, `starmap` will unpack it into the arguments of the function.

```python
def bar(x, y):
    print(f"Starting bar({x}, {y})")
    time.sleep(2)
    return x + y

with mp.Pool(processes=2) as pool:
    pool.starmap(bar, [(1, 2), (3, 4), (5, 6)])
```
```
Starting bar(1, 2)
Starting bar(3, 4)
Starting bar(5, 6)
---
Runtime: 6.0 seconds
```

`starmap` blocks. The async variant `starmap_async` is also available and do the exact thing that you would expect.


## `torch.multiprocessing`
The official documentation for `torch.multiprocessing` is [here](https://pytorch.org/docs/stable/multiprocessing.html). Also checkout the best practices [documentation](https://pytorch.org/docs/stable/notes/multiprocessing.html).

`torch.multiprocessing` is a wrapper of `multiprocessing` with extra functionalities, which API is fully compatible with the original module, so we can use it as a drop-in replacement. Let's try running an example from the previous section, but using `torch.multiprocessing`:

```python
import torch.multiprocessing as mp
import time

def foo(x):
    print(f"Starting foo({x})")
    time.sleep(2)
    return x

with mp.Pool(processes=2) as pool:
    result = pool.map(foo, range(10), chunksize=None)
    print(result)
```
```
Starting foo(0)
Starting foo(2)
Starting foo(1)
Starting foo(3)
Starting foo(4)
Starting foo(6)
Starting foo(5)
Starting foo(7)
Starting foo(8)
Starting foo(9)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
---
Runtime: 12.0 seconds
```
No difference from the previous example!

`multiprocessing` supports 3 process start methods: *fork* (default on Unix), *spawn* (default on Windows and MacOS), and *forkserver*. To use CUDA in subprocesses, one must use either *forkserver* or *spawn*. The start method should be set once by using `set_start_method()` in the `if __name__ == '__main__'` clause of the main module:
```python
import torch.multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('forkserver')
    ...
```

### Sharing tensors
`torch` makes use of 2 sharing strategies for CPU tensors: file descriptor (default) and file system.

It is recommended to use the *queue* strategy above to share tensors between processes. The tensors must be in shared memory, and they will be automatically moved to shared memory once `Queue.put(tensor)` is called if they are not already. `Queue.get()` returns a handle to the tensor in shared memory.

To manually move a tensor to shared memory, we can use `Tensor.share_memory_()`. This is a no-op if the tensor is already in shared memory, or if the tensor is a CUDA tensor. For `nn.Module`, we can move the module to shared memory by calling `.share_memory()`.

To check if a tensor is in shared memory, we can use `Tensor.is_shared()`.


```python
import torch.multiprocessing as mp
import time

mat = torch.randn((200, 200))
print(mat.is_shared())

queue = mp.Queue()
q.put(a)
print(a.is_shared())
```
```
False
True
---
Runtime: 0.0 seconds
```
Again, when we put a tensor into the queue, it is automatically moved to shared memory, that is why the second check returns `True`.

Note that if `Tensor.grad` is not `None`, it is also shared.

If the provider process exits while its tensor is still in a shared queue, attempts to get the tensor will raise an exception.

```python
import torch
import torch.multiprocessing as mp
import time

def foo(q):
    q.put(torch.randn(20, 20))
    q.put(torch.randn(10, 10))
    time.sleep(3)

def bar(q):
    t1 = q.get()
    print(f"Received {t1.size()}")
    time.sleep(4)
    t2 = q.get()
    print(f"Received {t2.size()}")

if __name__ == "__main__":
    mp.set_start_method('spawn')

    queue = mp.Queue()
    p1 = mp.Process(target=foo, args=(queue,))
    p2 = mp.Process(target=bar, args=(queue,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
```
```
Received torch.Size([20, 20])
Process Process-2:
Traceback (most recent call last):
  ...
  File "/home/term1nal/miniconda3/envs/ML/lib/python3.9/multiprocessing/connection.py", line 635, in SocketClient
    s.connect(address)
ConnectionRefusedError: [Errno 111] Connection refused
---
Runtime: 4.0 seconds
```
The first tensor is consumed while the provider process (running `foo`) is still alive. The second tensor is consumed when the provider process already exited, thus raising an error.

So make sure you consume the tensors in queue before the provider process exits, or employ some waiting mechanism on the provider side.

### Sharing CUDA tensors
It is basically the same as above, but must be handled with a bit more care.

CUDA tensors always use the CUDA API, and that is the only mechanism through which CUDA tensors can be shared. `Tensor.share_memory_()` is a no-op for CUDA tensors.

Unlike CPU tensors, it is required to keep the provider running as long as any consumer processes have references to a CUDA tensor. Once the consumer is done with the tensor, it should explicitly call `del` to release the memory. The following example is a bad practice:

```python
import torch
import torch.multiprocessing as mp
import time

def foo(q):
    q.put(torch.randn(20, 20).cuda())
    time.sleep(2)

def bar(q):
    tensor = q.get()
    time.sleep(2) #  delibrately sleep to make sure that foo is done
    print(f"Received {tensor.size()}")

if __name__ == "__main__":
    mp.set_start_method('spawn')

    queue = mp.Queue()
    p1 = mp.Process(target=foo, args=(queue,))
    p2 = mp.Process(target=bar, args=(queue,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
```
```
[W CudaIPCTypes.cpp:15] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]
Received torch.Size([20, 20])
```

### `spawn`
Creating multiple processes is hideous. If we want to start multiple processes running a function, we can do it like this:
```python
import torch.multiprocessing as mp

def foo():
    pass

if __name__ == "__main__":
    num_proc = 4
    processes = [mp.Process(target=foo)]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()
```
The problem lies in the *join* part. If the first process does not terminate, the termination of others will go unnoticed; and there are no facilities for error propagation.

`spawn` takes care of error propagation, out of order termination, and will actively terminate processes upon detecting an error in one of them.
```python
import torch.multiprocessing as mp

def foo(idx):
    pass

if __name__ == "__main__":
    mp.spawn(foo, args=(), nprocs=4, join=True)
```
The function `fn` passed to `spawn` (`foo` in this case) will be called as `fn(idx, *args)`, where `idx` is the index of the process.
```

## Closing remarks
The knowledge covered in this post should familiarize you with basic multiprocessing in Python/PyTorch. Checkout the next [post](https://www.youtube.com/watch?v=dQw4w9WgXcQ) of the series where we will discuss distributed communication in PyTorch.
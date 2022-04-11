---
title: "Multiprocessing in PyTorch"
excerpt: "Introduction to `multiprocessing` and `torch.multiprocessing`"
header:
  teaser: "assets/images/torch.png"
tags: 
  - python
  - pytorch
---
This post gives a gentle introduction to multiprocessing in Python and PyTorch, using the `torch.multiprocessing` module.

## Introduction
Multiprocessing is process-level parallelism, in the sense that each spawned process is allocated seperate memory and resources. In Python, in many cases, multiprocessing is used to bypass the infamous GIL, which is a global lock that prevents the interpreter from running any other code while it is executing a single thread.

## `multiprocessing`
In native Python, multiprocessing is achieved by using the `multiprocessing` module. 

The official documentation of `multiprocessing` is [here](https://docs.python.org/3/library/multiprocessing.html), and it is great! In this tutorial we will only cover 
some of the most important and relevant features of the module; for more details, please refer to the official documentation.

### Pool
`multiprocessing.Pool` creates a *pool of processes*, each of which is allocated a separate memory space. It is a context manager, so it can be used in a `with` statement.

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
`apply_async` is non-blocking and will return a `AsyncResult` object immediately. We can then use `get` to get the result of the task.

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
In some cases the second function may be completed before the first one, and correspondingly, the callback for the second function will be called before the callback for the first function.

#### `map` and `starmap`
gg

### Process
asdas

## `torch.multiprocessing`
PyTorch provides a wrapper of `multiprocessing` named `torch.multiprocessing`, which API is fully compatible with the original module.
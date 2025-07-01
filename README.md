# Hasktorch with Type-Safe Tensors

This is a pure Haskell tensor library inspired by PyTorch. It provides type-safe N-dimensional tensors, broadcasting, neural network building blocks, and more, all with type-level shape tracking.

<br>

> [Note]
> This is a toy library created as final project for the "Advanced functional programming" course at the University of Warsaw.

## Why Type-Level Shapes?

In mainstream frameworks like PyTorch, shape mistakes (mismatched dimensions, invalid broadcasting, bad matrix multiplies) are very common and only show up at _runtime_ - sometimes deep in training, causing hours of wasted debugging.

**With type-level shapes, your compiler checks shape correctness for you.**  
Shape mismatches are _type errors_, not runtime surprises.

#### Example:

When defining tensors, you specify their shapes as types:

```haskell
t1 :: Tensor '["DIM1", "DIM2"]
t2 :: Tensor '["DIM2", "DIM3"]
```

Observe that dimensions are named. The operations you perform on these tensors will be type-checked against their shapes, ensuring that operations like addition, multiplication, and indexing are valid.

Moreover, shapes will propagate through operations, so if you multiply tensor `["DIM1", "DIM2"]` with `["DIM2", "DIM3"]`, the result will be inferred as `["DIM1", "DIM3"]`.

<br>

> [NOTE]
> There is alternative way to implement type-safe tensors, and it is to store exact dimensions in the type system as natural numbers. There are existing libraries that do this so check them out if you are interested.

## Features

- N-dimensional tensors with type-level shapes
- Broadcasting
- Batched matrix multiplication and basic math
- Neural network primitives (Linear, ReLU, Softmax)
- Safe slicing, reshaping, and shape manipulation

---

## API Preview

**Tensor creation**

```haskell
-- From flat list
fromList [2, 3] [1.0,2.0,3.0,4.0,5.0,6.0] :: DoubleTensor '["a", "b"]

-- From nested lists (type-checked dimensions)
fromNested2 [[1.0, 2.0], [3.0, 4.0]] :: DoubleTensor '["x", "y"]

-- Scalar
fromScalar 2.0 :: DoubleTensor '["dummy"]

-- Zeros and ones
zeros [2,3]       -- :: DoubleTensor s
ones [4,4]        -- :: DoubleTensor s
```

**Math & Operations**

```haskell
t1 +. t2                -- Elementwise addition (broadcasted)
t1 -. t2                -- Subtraction
t1 *. t2                -- Elementwise multiplication (broadcasted)
t1 @. t2                -- Matrix multiplication (with batching if >2D)

relu t                  -- ReLU activation
softmax t               -- Softmax (last axis)

reshapeUnsafe t [4,2]   -- Reshape (same total size, unsafe)
unsqueeze @0 t          -- Add new axis at position 0
squeeze @1 t            -- Remove axis at position 1
swapaxes @0 @1 t        -- Swap axes 0 and 1

get t (0 :~ 1 :~ LNil)     -- Get element at specific indices
set t (0 :~ 1 :~ LNil) 5.0 -- Set element at specific indices

-- Pytorch equivalent: x[:, 1, 0:2]
slice t (All :~ Single 1 :~ Range 0 2) -- Slice tensor along specified axes

reduce V.sum t (Proxy @0 :- INil)
reduce V.maximum t (Proxy @1 :- INil)
```

**Pytorch-like Modules**

```haskell
fc = Linear w b
out = linear fc input
```

## Usage

To build, run, and test:

```sh
make        # build the library
make run    # run the main example
make test   # run the test suite
```

## Additional Information

`Main.hs` contains a simple example demonstrating MNIST digit classification using a small neural network.

To run `Main.hs`, you need to first train a model using pytorch framework and export it's weights to files. The code for model and exporting weights is in `pytorch` directory.

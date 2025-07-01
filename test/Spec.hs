module Main (main) where

import Control.DeepSeq (force)
import Control.Exception (evaluate)
import Data.Proxy (Proxy (..))
import Data.Vector qualified as V
import Test.Hspec
import Torch.Tensor
import Torch.Tensor.Op
import TypeSpec qualified

epsilon :: Double
epsilon = 1e-6

shouldBeClose :: (Ord a, Num a, Show a) => a -> a -> a -> Expectation
shouldBeClose x y epsilon = abs (x - y) `shouldSatisfy` (< epsilon)

vectorShouldBeClose :: (Ord a, Num a, Show a) => V.Vector a -> V.Vector a -> a -> Expectation
vectorShouldBeClose v1 v2 epsilon = do
  V.length v1 `shouldBe` V.length v2
  V.zipWithM_ (\x y -> shouldBeClose x y epsilon) v1 v2

doubleVectorShouldBeClose :: V.Vector Double -> V.Vector Double -> Expectation
doubleVectorShouldBeClose v1 v2 = vectorShouldBeClose v1 v2 epsilon

mean :: (Fractional a) => V.Vector a -> a
mean v = V.sum v / fromIntegral (V.length v)

main :: IO ()
main = hspec $ do
  describe "Tensor creation" $ do
    it "creates a DoubleTensor" $ do
      let t = fromList [2, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: DoubleTensor '["a", "b"]
      shape t `shouldBe` [2, 3]
      array t `shouldBe` V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    it "creates a FloatTensor" $ do
      let t = fromList [2, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: FloatTensor '["a", "b"]
      shape t `shouldBe` [2, 3]
      array t `shouldBe` V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    it "creates an IntTensor" $ do
      let t = fromList [2, 3] [1, 2, 3, 4, 5, 6] :: IntTensor '["a", "b"]
      shape t `shouldBe` [2, 3]
      array t `shouldBe` V.fromList [1, 2, 3, 4, 5, 6]

    it "throws when empty shape list is provided" $ do
      evaluate (fromList [] [1.0, 2.0, 3.0]) `shouldThrow` anyException

    it "throws when shape and array length mismatch" $ do
      evaluate (fromList [2, 3] [1.0, 2.0]) `shouldThrow` anyException

    it "creates a tensor from nested lists" $ do
      let t = fromNested2 [[1.0, 2.0], [3.0, 4.0]] :: DoubleTensor '["a", "b"]
      shape t `shouldBe` [2, 2]
      array t `shouldBe` V.fromList [1.0, 2.0, 3.0, 4.0]

      let t2 = fromNested3 [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]] :: DoubleTensor '["batch", "a", "b"]
      shape t2 `shouldBe` [2, 2, 2]
      array t2 `shouldBe` V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

      let t3 = fromNested4 [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]] :: DoubleTensor '["batch", "a", "b", "c"]
      shape t3 `shouldBe` [2, 2, 2, 2]
      array t3 `shouldBe` V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]

    it "throws when nested lists are empty" $ do
      evaluate (fromNested2 []) `shouldThrow` anyException

    it "throws when nested lists have different lengths" $ do
      evaluate (fromNested2 [[1.0, 2.0], [3.0]]) `shouldThrow` anyException

  describe "Tensor operations" $ do
    it "performs same shape operations" $ do
      let t1 = fromList [2, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: DoubleTensor '["a", "b"]
          t2 = fromList [2, 3] [6.0, 5.0, 4.0, 3.0, 2.0, 1.0] :: DoubleTensor '["a", "b"]
          tAdd :: DoubleTensor '["a", "b"] = t1 +. t2
          tSub = t1 -. t2
          tMul = t1 *. t2
          tDiv = t1 /. t2
      shape tAdd `shouldBe` [2, 3]
      array tAdd `shouldBe` V.fromList [7.0, 7.0, 7.0, 7.0, 7.0, 7.0]
      shape tSub `shouldBe` [2, 3]
      array tSub `shouldBe` V.fromList [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]
      shape tMul `shouldBe` [2, 3]
      array tMul `shouldBe` V.fromList [6.0, 10.0, 12.0, 12.0, 10.0, 6.0]
      shape tDiv `shouldBe` [2, 3]
      array tDiv `doubleVectorShouldBeClose` V.fromList [0.16666667, 0.4, 0.75, 1.3333333, 2.5, 6.0]

  it "performs broadcasting operations" $ do
    let t1 = fromList [2, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: DoubleTensor '["a", "b"]
        t2 = fromList [3] [1.0, 2.0, 3.0] :: DoubleTensor '["b"]
        tAdd :: DoubleTensor '["a", "b"] = t1 +. t2
        tSub = t1 -. t2
        tMul = t1 *. t2
        tDiv = t1 /. t2
    shape tAdd `shouldBe` [2, 3]
    array tAdd `shouldBe` V.fromList [2.0, 4.0, 6.0, 5.0, 7.0, 9.0]
    shape tSub `shouldBe` [2, 3]
    array tSub `shouldBe` V.fromList [0.0, 0.0, 0.0, 3.0, 3.0, 3.0]
    shape tMul `shouldBe` [2, 3]
    array tMul `shouldBe` V.fromList [1.0, 4.0, 9.0, 4.0, 10.0, 18.0]
    shape tDiv `shouldBe` [2, 3]
    array tDiv `doubleVectorShouldBeClose` V.fromList [1.0, 1.0, 1.0, 4.0, 2.5, 2.0]
    isClose tAdd (t2 +. t1) epsilon `shouldBe` True
    isClose tMul (t2 *. t1) epsilon `shouldBe` True

    let tScalar = fromScalar 2.0 :: DoubleTensor '["b"]
        tAddScalar = t1 +. tScalar
        tSubScalar = t1 -. tScalar
        tMulScalar = t1 *. tScalar
        tDivScalar = t1 /. tScalar
    shape tAddScalar `shouldBe` [2, 3]
    array tAddScalar `shouldBe` V.fromList [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    shape tSubScalar `shouldBe` [2, 3]
    array tSubScalar `shouldBe` V.fromList [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    shape tMulScalar `shouldBe` [2, 3]
    array tMulScalar `shouldBe` V.fromList [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    shape tDivScalar `shouldBe` [2, 3]
    array tDivScalar `doubleVectorShouldBeClose` V.fromList [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

  it "performs matrix multiplication" $ do
    let t1 = fromNested2 [[1.0, 2.0], [3.0, 4.0]] :: DoubleTensor '["a", "b"]
        t2 = fromNested2 [[5.0, 6.0], [7.0, 8.0]] :: DoubleTensor '["b", "c"]
        tMatMul = t1 @. t2
    shape tMatMul `shouldBe` [2, 2]
    array tMatMul `doubleVectorShouldBeClose` V.fromList [19.0, 22.0, 43.0, 50.0]

    let t3 = fromNested2 [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] :: DoubleTensor '["a", "b"]
        t4 = fromNested2 [[7.0], [8.0], [9.0]] :: DoubleTensor '["b", "c"]
        tMatMul2 = t3 @. t4
    shape tMatMul2 `shouldBe` [2, 1]
    array tMatMul2 `doubleVectorShouldBeClose` V.fromList [50.0, 122.0]

    let t5 = fromNested2 [[2.0, 1.0], [5.0, 3.0]] :: DoubleTensor '["a", "b"]
        t6 = fromNested2 [[3.0, -1.0], [-5.0, 2.0]] :: DoubleTensor '["b", "a"]
        tMatMul3 = t5 @. t6
    shape tMatMul3 `shouldBe` [2, 2]
    array tMatMul3 `doubleVectorShouldBeClose` V.fromList [1.0, 0.0, 0.0, 1.0]

  it "performs batched matrix multiplication" $ do
    let t1 = fromNested3 [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]] :: DoubleTensor '["batch", "a", "b"]
        t2 = fromNested3 [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]] :: DoubleTensor '["batch", "b", "c"]
        tBatchedMatMul = t1 @. t2
    shape tBatchedMatMul `shouldBe` [2, 2, 2]
    array tBatchedMatMul `doubleVectorShouldBeClose` V.fromList [31.0, 34.0, 71.0, 78.0, 155.0, 166.0, 211.0, 226.0]

  it "performs broadcasted matrix multiplication" $ do
    let t1 = fromNested3 [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]] :: DoubleTensor '["batch", "a", "b"]
        t2 = fromNested2 [[13.0, 14.0], [15.0, 16.0]] :: DoubleTensor '["b", "c"]
        tBatchedMatMul = t1 @. t2
    shape tBatchedMatMul `shouldBe` [2, 2, 2]
    print tBatchedMatMul
    array tBatchedMatMul `doubleVectorShouldBeClose` V.fromList [43.0, 46.0, 99.0, 106.0, 155.0, 166.0, 211.0, 226.0]

  it "gets and sets elements" $ do
    let t = fromList [2, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: DoubleTensor '["a", "b"]
    get t (0 :~ 1 :~ LNil) `shouldBe` 2.0
    let t' = set t (0 :~ 1 :~ LNil) 10.0
    get t' (0 :~ 1 :~ LNil) `shouldBe` 10.0
    array t' `shouldBe` V.fromList [1.0, 10.0, 3.0, 4.0, 5.0, 6.0]

  it "squeezes and unsqueezes dimensions" $ do
    let t = fromList [3] [1.0, 2.0, 3.0] :: DoubleTensor '["a"]
    let unsqT = unsqueeze @0 t
    shape unsqT `shouldBe` [1, 3]
    let unsqT2 = unsqueeze @2 unsqT
    shape unsqT2 `shouldBe` [1, 3, 1]
    let sqT = squeeze @2 unsqT2
    shape sqT `shouldBe` [1, 3]
    let sqT2 = squeeze @0 sqT
    shape sqT2 `shouldBe` [3]

  it "unsqueezes to a specific shape" $ do
    let t1 = fromList [3] [1.0, 2.0, 3.0] :: DoubleTensor '["a"]
    let t2 = fromList [2, 3] [1.0, 2.0, 3.0, 1.0, 2.0, 3.0] :: DoubleTensor '["b", "a"]
    let unsqT = unsqueezeTo t1 t2
    shape unsqT `shouldBe` [1, 3]

    let t3 = unsqueeze @0 t2
    let unsqT2 = unsqueezeTo t1 t3
    shape unsqT2 `shouldBe` [1, 1, 3]

  it "slices tensors" $ do
    let t = fromList [2, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: DoubleTensor '["a", "b"]
    let sliced = slice t (Single 1 :~ Range 1 2 :~ LNil)
    shape sliced `shouldBe` [1, 1]
    array sliced `doubleVectorShouldBeClose` V.fromList [5.0]

    let sliced2 = slice t (All :~ Single 1 :~ LNil)
    shape sliced2 `shouldBe` [2, 1]
    array sliced2 `doubleVectorShouldBeClose` V.fromList [2.0, 5.0]

    let sliced3 = slice t (Single 0 :~ All :~ LNil)
    shape sliced3 `shouldBe` [1, 3]
    array sliced3 `doubleVectorShouldBeClose` V.fromList [1.0, 2.0, 3.0]

  it "broadcasts tensors" $ do
    let t2 = fromList [2, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: DoubleTensor '["a", "b"]
        t1 = fromList [3] [10.0, 20.0, 30.0] :: DoubleTensor '["b"]
        tBroadcasted = broadcastTensorTo t1 t2
    shape tBroadcasted `shouldBe` [2, 3]
    array tBroadcasted `doubleVectorShouldBeClose` V.fromList [10.0, 20.0, 30.0, 10.0, 20.0, 30.0]

  it "reshapes tensors" $ do
    let t = fromList [2, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: DoubleTensor '["a", "b"]
    let reshaped = reshapeUnsafe t [3, 2]
    shape reshaped `shouldBe` [3, 2]
    array reshaped `doubleVectorShouldBeClose` V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    evaluate (force $ reshapeUnsafe t [3]) `shouldThrow` anyErrorCall
    evaluate (force $ reshapeUnsafe t [4]) `shouldThrow` anyErrorCall

    let reshaped2 = reshapeUnsafe t [-1]
    shape reshaped2 `shouldBe` [6]

    evaluate (force $ reshapeUnsafe t [0]) `shouldThrow` anyErrorCall
    evaluate (force $ reshapeUnsafe t [-1, 12]) `shouldThrow` anyErrorCall
    evaluate (force $ reshapeUnsafe t [4, -1]) `shouldThrow` anyErrorCall

    let reshaped3 = reshapeUnsafe (zeros [5, 3, 2]) [5, -1]
    shape reshaped3 `shouldBe` [5, 6]

  it "swaps axes" $ do
    let t = fromList [2, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: DoubleTensor '["a", "b"]
    let swapped :: DoubleTensor '["b", "a"] = swapaxes @0 @1 t
    shape swapped `shouldBe` [3, 2]
    array swapped `doubleVectorShouldBeClose` V.fromList [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]

    let swapped2 :: DoubleTensor '["a", "b"] = swapaxes @1 @0 swapped
    shape swapped2 `shouldBe` [2, 3]
    array swapped2 `doubleVectorShouldBeClose` V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

  it "reduces tensors" $ do
    let t = fromList [2, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: DoubleTensor '["a", "b"]
        reducedSum :: DoubleTensor '[Any] = reduce V.sum t (Proxy @0 :- Proxy @1 :- INil)
    shape reducedSum `shouldBe` [1]
    array reducedSum `doubleVectorShouldBeClose` V.fromList [21.0]

    let reducesSum2 :: DoubleTensor '["b"] = reduce V.sum t (Proxy @0 :- INil)
    shape reducesSum2 `shouldBe` [3]
    array reducesSum2 `doubleVectorShouldBeClose` V.fromList [5.0, 7.0, 9.0]

    let reducedSum3 :: DoubleTensor '["a"] = reduce V.sum t (Proxy @1 :- INil)
    shape reducedSum3 `shouldBe` [2]
    array reducedSum3 `doubleVectorShouldBeClose` V.fromList [6.0, 15.0]

    let reducedMean :: DoubleTensor '[Any] = reduce mean t (Proxy @0 :- Proxy @1 :- INil)
    shape reducedMean `shouldBe` [1]
    array reducedMean `doubleVectorShouldBeClose` V.fromList [3.5]
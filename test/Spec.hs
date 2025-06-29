module Main (main) where

import Control.Exception (evaluate)
import Data.Vector qualified as V
import Test.Hspec
import Torch.Tensor
import TypeSpec qualified

epsilon :: Double
epsilon = 1e-6

shouldBeClose :: (Ord a, Num a, Show a) => a -> a -> a -> Expectation
shouldBeClose x y epsilon = abs (x - y) `shouldSatisfy` (< epsilon)

vectorShouldBeClose :: (Ord a, Num a, Show a) => V.Vector a -> V.Vector a -> a -> Expectation
vectorShouldBeClose v1 v2 epsilon = do
  V.length v1 `shouldBe` V.length v2
  V.zipWithM_ (\x y -> shouldBeClose x y epsilon) v1 v2 -- TODO: check this

doubleVectorShouldBeClose :: V.Vector Double -> V.Vector Double -> Expectation
doubleVectorShouldBeClose v1 v2 = vectorShouldBeClose v1 v2 epsilon

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

    it "throws when nested lists are empty" $ do
      evaluate (fromNested2 []) `shouldThrow` anyException

    it "throws when nested lists have different lengths" $ do
      evaluate (fromNested2 [[1.0, 2.0], [3.0]]) `shouldThrow` anyException

  describe "Tensor operations" $ do
    it "performs same shape operations" $ do
      let t1 = fromList [2, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: DoubleTensor '["a", "b"]
          t2 = fromList [2, 3] [6.0, 5.0, 4.0, 3.0, 2.0, 1.0] :: DoubleTensor '["a", "b"]
          tAdd = t1 +. t2
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

-- it "performs broadcasting operations" $ do
--   let t1 = fromList [2, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: DoubleTensor '["a", "b"]
--       t2 = fromList [3] [1.0, 2.0, 3.0] :: DoubleTensor '["b"]
--       tAdd = t1 +. t2
--       tSub = t1 -. t2
--       tMul = t1 *. t2
--       tDiv = t1 /. t2
--   shape tAdd `shouldBe` [2, 3]
--   array tAdd `shouldBe` V.fromList [2.0, 4.0, 6.0, 5.0, 7.0, 9.0]
--   shape tSub `shouldBe` [2, 3]
--   array tSub `shouldBe` V.fromList [0.0, 0.0, 0.0, 3.0, 3.0, 3.0]
--   shape tMul `shouldBe` [2, 3]
--   array tMul `shouldBe` V.fromList [1.0, 4.0, 9.0, 4.0, 10.0, 18.0]
--   shape tDiv `shouldBe` [2, 3]
--   array tDiv `doubleVectorShouldBeClose` V.fromList [1.0, 1.0, 1.0, 4.0, 2.5, 2.0]

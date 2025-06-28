module Tensor.Op where

import Data.Vector (Vector, (!), (//))
import Data.Vector qualified as V
import Tensor.Internal
import Tensor.Types

fromList :: [Int] -> [a] -> Tensor s a
fromList shape array
  | any (<= 0) shape =
      error ("Shape must contain only positive integers: " ++ show shape)
  | length array /= expectedShape =
      error ("Array cannot be reshaped: array length = " ++ show (length array) ++ ", expected length = " ++ show expectedShape)
  | otherwise =
      Tensor {shape = shape, array = V.fromList array}
  where
    expectedShape = product shape

sameShapeOp :: (Compatible s1 s2 ~ 'True) => (a -> a -> a) -> Tensor s1 a -> Tensor s2 a -> Tensor s1 a
sameShapeOp op (Tensor {shape = s1, array = a1}) (Tensor {shape = s2, array = a2}) =
  case broadcastShapes s1 s2 of
    Nothing -> error ("Cannot broadcast shapes: " ++ show s1 ++ " and " ++ show s2)
    Just bcShape ->
      let a1' = broadcastArray a1 s1 bcShape
          a2' = broadcastArray a2 s2 bcShape
       in Tensor {shape = bcShape, array = V.zipWith op a1' a2'}

tensorAdd :: (Compatible s1 s2 ~ 'True, Num a) => Tensor s1 a -> Tensor s2 a -> Tensor s1 a
tensorAdd = sameShapeOp (+)

tensorSub :: (Compatible s1 s2 ~ 'True, Num a) => Tensor s1 a -> Tensor s2 a -> Tensor s1 a
tensorSub = sameShapeOp (-)

tensorMul :: (Compatible s1 s2 ~ 'True, Num a) => Tensor s1 a -> Tensor s2 a -> Tensor s1 a
tensorMul = sameShapeOp (*)

tensorDiv :: (Compatible s1 s2 ~ 'True, Fractional a) => Tensor s1 a -> Tensor s2 a -> Tensor s1 a
tensorDiv = sameShapeOp (/)

infixl 6 +.

infixl 6 -.

infixl 7 *.

infixl 7 /.

(+.) :: (Compatible s1 s2 ~ 'True, Num a) => Tensor s1 a -> Tensor s2 a -> Tensor s1 a
(+.) = tensorAdd

(-.) :: (Compatible s1 s2 ~ 'True, Num a) => Tensor s1 a -> Tensor s2 a -> Tensor s1 a
(-.) = tensorSub

(*.) :: (Compatible s1 s2 ~ 'True, Num a) => Tensor s1 a -> Tensor s2 a -> Tensor s1 a
(*.) = tensorMul

(/.) :: (Compatible s1 s2 ~ 'True, Fractional a) => Tensor s1 a -> Tensor s2 a -> Tensor s1 a
(/.) = tensorDiv

indexUnsafe :: Tensor s a -> [Int] -> a
indexUnsafe (Tensor {shape = s, array = a}) indices =
  indexFromStride a strides indices
  where
    strides = stridesFromShape s

index :: forall s a n. (Length s ~ n) => Tensor s a -> LList n Int -> a
index t l = indexUnsafe t (toList l)

setUnsafe :: Tensor s a -> [Int] -> a -> Tensor s a
setUnsafe (Tensor {shape = s, array = a}) indices value =
  let pos = indicesToPos (stridesFromShape s) indices
      newArray = a // [(pos, value)]
   in Tensor {shape = s, array = newArray}

set :: forall s a n. (Length s ~ n) => Tensor s a -> LList n Int -> a -> Tensor s a
set t l value = setUnsafe t (toList l) value

broadcastTensorTo :: Tensor s1 a -> [Int] -> Tensor s1 a
broadcastTensorTo (Tensor {shape = s1, array = a1}) shape =
  case broadcastShapes s1 shape of
    Nothing -> error ("Cannot broadcast shape: " ++ show s1 ++ " to " ++ show shape)
    Just s2
      | shape == s2 -> Tensor {shape = shape, array = broadcastArray a1 s1 shape}
      | otherwise -> error ("Given shape: " ++ show shape ++ " has unit dimensions that would need to be broadcasted to input tensor shape")

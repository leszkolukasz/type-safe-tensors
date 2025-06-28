module Tensor
  ( Tensor (..),
    DoubleTensor,
    Any,
    Shape,
    Compatible,
    fromList,
    index,
    tensorAdd,
    tensorSub,
    tensorMul,
    tensorDiv,
    (+.),
    (-.),
    (*.),
    (/.),
  )
where

import Data.Kind (Type)
import Data.Vector (Vector, (!), (//))
import Data.Vector qualified as V
import GHC.TypeLits (Nat, Symbol)
import Debug.Trace (trace)

type Any = "Any"

type Shape = [Symbol]

type Tensor :: Shape -> Type -> Type
data Tensor s a where
  Tensor :: {shape :: [Int], array :: Vector a} -> Tensor s a
  deriving (Show)

type DoubleTensor s = Tensor s Double

instance Functor (Tensor s) where
  fmap f (Tensor {shape = s, array = a}) = Tensor {shape = s, array = fmap f a}

type family Compatible (s1 :: Shape) (s2 :: Shape) :: Bool where
  Compatible '[] '[] = 'True
  Compatible (n : s1) (n : s2) = Compatible s1 s2
  Compatible (Any : s1) (_ : s2) = Compatible s1 s2
  Compatible (_ : s1) (Any : s2) = Compatible s1 s2
  Compatible _ _ = 'False

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

indexFromStride :: Vector a -> [Int] -> [Int] -> a
indexFromStride arr strides indices =
  let pos = sum $ zipWith (*) indices strides
   in arr ! pos

stridesFromShape :: [Int] -> [Int]
stridesFromShape [a] = [1]
stridesFromShape (x : xs) = product xs : stridesFromShape xs

stridesForBroadcast :: [Int] -> [Int] -> [Int]
stridesForBroadcast oldShape broadcastShape =
  let oldStrides = stridesFromShape oldShape
      newStrides = stridesFromShape broadcastShape
   in zipWith (\sd sh -> if fst sh == 1 && snd sh /= 1 then 0 else sd) oldStrides (zip oldShape broadcastShape)

indicesToPos :: [Int] -> [Int] -> Int
indicesToPos strides indices =
  sum $ zipWith (*) indices strides

posToIndices :: [Int] -> Int -> [Int]
posToIndices strides pos = go strides pos []
  where
    go [] _ acc = reverse acc
    go (s : ss) p acc =
      let (q, r) = p `divMod` s
       in go ss r (q : acc)

index :: Tensor s a -> [Int] -> a
index (Tensor {shape = s, array = a}) indices =
  indexFromStride a strides indices
  where
    strides = stridesFromShape s

set :: Tensor s a -> [Int] -> a -> Tensor s a
set (Tensor {shape = s, array = a}) indices value =
  let pos = indicesToPos (stridesFromShape s) indices
      newArray = a // [(pos, value)]
   in Tensor {shape = s, array = newArray}

broadcastArray :: Vector a -> [Int] -> [Int] -> Vector a
broadcastArray arr oldShape newShape
  | oldShape == newShape = arr
  | otherwise =
      let broadcastStrides = stridesForBroadcast oldShape newShape
          totalElements = product newShape
          newArray = V.generate totalElements $ \i ->
            let indices = posToIndices (stridesFromShape newShape) i in
               indexFromStride arr broadcastStrides indices
       in newArray

-- Finds common shape for broadcasting of both arrays
broadcastShapes :: [Int] -> [Int] -> Maybe [Int]
broadcastShapes [] [] = Just []
broadcastShapes [] ys = Nothing
broadcastShapes xs [] = Nothing
broadcastShapes (x : xs) (y : ys)
  | x == y = (x :) <$> broadcastShapes xs ys
  | x == 1 = (y :) <$> broadcastShapes xs ys
  | y == 1 = (x :) <$> broadcastShapes xs ys
  | otherwise = Nothing

broadcastTensorTo :: Tensor s1 a -> [Int] -> Tensor s1 a
broadcastTensorTo (Tensor {shape = s1, array = a1}) shape =
  case broadcastShapes s1 shape of
    Nothing -> error ("Cannot broadcast shape: " ++ show s1 ++ " to " ++ show shape)
    Just s2
      | shape == s2 -> Tensor {shape = shape, array = broadcastArray a1 s1 shape}
      | otherwise -> error ("Given shape: " ++ show shape ++ " has unit dimensions that would need to be broadcasted to input tensor shape")

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
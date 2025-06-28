module Tensor
  ( Tensor (..),
    DoubleTensor,
    Any,
    Shape,
    Compatible,
    fromList,
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
import GHC.TypeLits (Nat, Symbol)

type Any = "Any"

type Shape = [Symbol]

type Tensor :: Shape -> Type -> Type
data Tensor s a where
  Tensor :: {shape :: [Int], stride :: [Int], array :: [a]} -> Tensor s a
  deriving (Show)

type DoubleTensor s = Tensor s Double

instance Functor (Tensor s) where
  fmap f (Tensor {shape = s, stride = sd, array = a}) = Tensor {shape = s, stride = sd, array = map f a}

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
      Tensor {shape = shape, stride = strideFromShape shape, array = array}
  where
    expectedShape = product shape
    strideFromShape [x] = [1]
    strideFromShape (x:xs) = product xs : map strideFromShape xs

broadcastShape :: [Int] -> [Int] -> Maybe [Int]
broadcastShape [] [] = Just []
broadcastShape [] ys = Nothing
broadcastShape xs [] = Nothing
broadcastShape (x:xs) (y:ys)
  | x == y = (x :) <$> broadcastShape xs ys
  | x == 1 = (y :) <$> broadcastShape xs ys
  | y == 1 = (x :) <$> broadcastShape xs ys
  | otherwise = Nothing

broadcastTensorTo :: Tensor s1 a -> Tensor s2 a -> Maybe (Tensor s3 a)
broadcastTensorTo from@(Tensor {shape = s1, array = a1}) to@(Tensor {shape = s2}) =
  case broadcastShape s1 s2 of
    Nothing -> Nothing
    Just s3 -> Just $ Tensor {shape = s3, stride = strideFromShape s1 s3, array = a1}
  where
    strideFromShape [] [] = []
    strideFromShape [] _ = error "Cannot broadcast empty shape to non-empty shape"
    strideFromShape _ [] = error "Cannot broadcast non-empty shape to empty shape"
    strideFromShape @old(x:xs) @new(y:ys)
      | x == 1 && y /= 1 = 0 : strideFromShape xs ys
      | otherwise = x : strideFromShape xs ys
    


sameShapeOp :: (Compatible s1 s2 ~ 'True) => (a -> a -> a) -> Tensor s1 a -> Tensor s2 a -> Tensor s1 a
sameShapeOp op (Tensor {shape = s1, array = a1}) (Tensor {shape = s2, array = a2}) =
  case broadcastShape s1 s2 of
    Nothing -> error ("Cannot broadcast shapes: " ++ show s1 ++ " and " ++ show s2)
    Just s3 -> Tensor {shape = s1, array = zipWith op a1 a2}

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
module Torch.Tensor.Op where

import Data.Maybe (fromJust, listToMaybe)
import Data.Proxy (Proxy (..))
import Data.Vector (Vector, (!), (//))
import Data.Vector qualified as V
import GHC.TypeLits (KnownNat, Nat, natVal, type (+), type (-), type (<=))
import Torch.Tensor.Internal
import Torch.Tensor.Types
import Torch.Utils ((|>))

-- TODO: do constructors like fromNested2, fromNested3 for inputs of type [[a]], [[[a]]] etc.

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

fromNested2 :: [[a]] -> Tensor [s1, s2] a
fromNested2 nested
  | null nested || any null nested =
      error "Input cannot be empty or contain empty lists"
  | otherwise =
      let shape = [length nested, length (head nested)]
          array = concat nested
       in if not (validateShape shape nested)
            then error ("Invalid shape: " ++ show shape ++ " for input")
            else fromList shape array
  where
    validateShape :: [Int] -> [[a]] -> Bool
    validateShape s arr =
      if any (/= s !! 1) (map length arr)
        then error "All inner lists must have the same length"
        else True

sameShapeOp :: (Broadcastable s1 s2 ~ 'True) => (a -> a -> a) -> Tensor s1 a -> Tensor s2 a -> Tensor (BroadcastUnion s1 s2) a
sameShapeOp op (Tensor {shape = s1, array = a1}) (Tensor {shape = s2, array = a2}) =
  let s1' = unsqueezeShapeToIfPossible s1 s2
      s2' = unsqueezeShapeToIfPossible s2 s1
   in case broadcastShapes s1' s2' of
        Nothing -> error ("Cannot broadcast shapes: " ++ show s1 ++ " and " ++ show s2)
        Just bcShape ->
          let a1' = broadcastArray a1 s1' bcShape
              a2' = broadcastArray a2 s2' bcShape
           in Tensor {shape = bcShape, array = V.zipWith op a1' a2'}

tensorAdd :: (Broadcastable s1 s2 ~ 'True, Num a) => Tensor s1 a -> Tensor s2 a -> Tensor (BroadcastUnion s1 s2) a
tensorAdd = sameShapeOp (+)

tensorSub :: (Broadcastable s1 s2 ~ 'True, Num a) => Tensor s1 a -> Tensor s2 a -> Tensor (BroadcastUnion s1 s2) a
tensorSub = sameShapeOp (-)

tensorMul :: (Broadcastable s1 s2 ~ 'True, Num a) => Tensor s1 a -> Tensor s2 a -> Tensor (BroadcastUnion s1 s2) a
tensorMul = sameShapeOp (*)

tensorDiv :: (Broadcastable s1 s2 ~ 'True, Fractional a) => Tensor s1 a -> Tensor s2 a -> Tensor (BroadcastUnion s1 s2) a
tensorDiv = sameShapeOp (/)

-- TODO: make it broadcastable
matmul ::
  forall s1 s2 l n1 n2 m1 m2 a.
  ( Broadcastable (DropLastN s1 2) (DropLastN s2 2) ~ 'True,
    SndToLast s1 ~ 'Just n1,
    Last s1 ~ 'Just n2,
    SndToLast s2 ~ 'Just m1,
    Last s2 ~ 'Just m2,
    Compatible '[n1] '[n2] ~ 'True,
    Num a
  ) =>
  Tensor s1 a ->
  Tensor s2 a ->
  Tensor (Concat (DropLastN s1 2) [n1, m2]) a
matmul t1@(Tensor {shape = s1, array = a1}) t2@(Tensor {shape = s2, array = a2}) =
  let s1' = unsqueezeShapeToIfPossible s1 s2
      s2' = unsqueezeShapeToIfPossible s2 s1
      prefix1 = take (length s1 - 2) s1'
      prefix2 = take (length s2 - 2) s2'
   in case broadcastShapes prefix1 prefix2 of
        Nothing -> error ("Cannot broadcast shapes: " ++ show s1 ++ " and " ++ show s2)
        Just bcShape ->
          let s1'' = bcShape ++ drop (length prefix1) s1'
              s2'' = bcShape ++ drop (length prefix2) s2'
              a1' = broadcastArray a1 s1 s1''
              a2' = broadcastArray a2 s2 s2''
              t1' = Tensor {shape = s1'', array = a1'}
              t2' = Tensor {shape = s2'', array = a2'}
              n1 = last (init s1'')
              m2 = last s2''
              newShape = prefix1 ++ [n1, m2]
              totalElements = product newShape
              newArray =
                V.generate totalElements $
                  \i ->
                    let indices = posToIndices (stridesFromShape newShape) i
                        rowIdx = indices !! (length indices - 2)
                        colIdx = last indices
                        slice1 = map Single (take (length s1'' - 2) indices) ++ [Single rowIdx, All]
                        slice2 = map Single (take (length s2'' - 2) indices) ++ [All, Single colIdx]
                        row = array $ sliceUsafe t1' slice1
                        col = array $ sliceUsafe t2' slice2
                        dotProduct = V.sum $ V.zipWith (*) row col
                     in dotProduct
           in Tensor {shape = newShape, array = newArray}

infixl 6 +.

infixl 6 -.

infixl 7 *.

infixl 7 /.

infixl 7 @.

(+.) :: (Broadcastable s1 s2 ~ 'True, Num a) => Tensor s1 a -> Tensor s2 a -> Tensor (BroadcastUnion s1 s2) a
(+.) = tensorAdd

(-.) :: (Broadcastable s1 s2 ~ 'True, Num a) => Tensor s1 a -> Tensor s2 a -> Tensor (BroadcastUnion s1 s2) a
(-.) = tensorSub

(*.) :: (Broadcastable s1 s2 ~ 'True, Num a) => Tensor s1 a -> Tensor s2 a -> Tensor (BroadcastUnion s1 s2) a
(*.) = tensorMul

(/.) :: (Broadcastable s1 s2 ~ 'True, Fractional a) => Tensor s1 a -> Tensor s2 a -> Tensor (BroadcastUnion s1 s2) a
(/.) = tensorDiv

(@.) ::
  forall s1 s2 l n1 n2 m1 m2 a.
  ( Broadcastable (DropLastN s1 2) (DropLastN s2 2) ~ 'True,
    SndToLast s1 ~ 'Just n1,
    Last s1 ~ 'Just n2,
    SndToLast s2 ~ 'Just m1,
    Last s2 ~ 'Just m2,
    Compatible '[n1] '[n2] ~ 'True,
    Num a
  ) =>
  Tensor s1 a ->
  Tensor s2 a ->
  Tensor (Concat (DropLastN s1 2) [n1, m2]) a
(@.) = matmul

getUnsafe :: Tensor s a -> [Int] -> a
getUnsafe (Tensor {shape = s, array = a}) indices =
  let normIndices = normalizeIndices s indices
   in indexFromStride a strides normIndices
  where
    strides = stridesFromShape s

get :: forall s a n. (Length s ~ n) => Tensor s a -> LList n Int -> a
get t l = getUnsafe t (toList l)

setUnsafe :: Tensor s a -> [Int] -> a -> Tensor s a
setUnsafe (Tensor {shape = s, array = a}) indices value =
  let normIndices = normalizeIndices s indices
      pos = indicesToPos (stridesFromShape s) normIndices
      newArray = a // [(pos, value)]
   in Tensor {shape = s, array = newArray}

set :: forall s a n. (Length s ~ n) => Tensor s a -> LList n Int -> a -> Tensor s a
set t l value = setUnsafe t (toList l) value

unsqueezeUnsafe :: Tensor s a -> Int -> Tensor s' a
unsqueezeUnsafe (Tensor {shape = s, array = a}) n =
  if n < 0 || n > length s
    then error ("Invalid dimension " ++ show n ++ " for shape " ++ show s)
    else
      let newShape = take n s ++ [1] ++ drop n s
       in Tensor {shape = newShape, array = a}

unsqueeze :: forall n s a. (n <= Length s, KnownNat n) => Tensor s a -> Tensor (Unsqueeze s n) a
unsqueeze t = unsqueezeUnsafe t (fromIntegral (natVal (Proxy @n)))

squeezeUnsafe :: Tensor s a -> Int -> Tensor s' a
squeezeUnsafe (Tensor {shape = s, array = a}) n =
  if s !! n /= 1
    then error ("Cannot squeeze dimension " ++ show n ++ " of shape " ++ show s ++ " because it is not 1")
    else
      let newShape = take n s ++ drop (n + 1) s
       in Tensor {shape = newShape, array = a}

squeeze :: forall n s a. (n + 1 <= Length s, KnownNat n) => Tensor s a -> Tensor (DropNth s n) a
squeeze t = squeezeUnsafe t (fromIntegral (natVal (Proxy @n)))

-- Unsqueeze a tensor to the same shape length as another tensor
unsqueezeTo ::
  forall n s1 s2 a l1 l2.
  ( l1 ~ Length s1,
    l2 ~ Length s2,
    -- KnownNat l1,
    -- KnownNat l2,
    l1 <= l2,
    Compatible (DropFirstN s2 (l2 - l1)) s1 ~ 'True
  ) =>
  Tensor s1 a ->
  Tensor s2 a ->
  -- Tensor (Prepend s1 (Length s2 - Length s1) Any) a
  Tensor s2 a
unsqueezeTo from@(Tensor {shape = s1, array = a1}) to@(Tensor {shape = s2}) =
  let n = length s2 - length s1
      newShape = replicate n 1 ++ s1
   in Tensor {shape = newShape, array = a1}

test :: DoubleTensor [Any, "Dim"]
test = unsqueeze @(0) (fromList [3] [4.0, 5.0, 6.0] :: DoubleTensor '["Dim"])

broadcastTensorTo :: Tensor s1 a -> Tensor s2 a -> Tensor s2 a
broadcastTensorTo (Tensor {shape = s1, array = a1}) (Tensor {shape = s2}) =
  case broadcastShapes s1 s2 of
    Nothing -> error ("Cannot broadcast shape: " ++ show s1 ++ " to " ++ show s2)
    Just s3
      | s2 == s3 -> Tensor {shape = s2, array = broadcastArray a1 s1 s2}
      | otherwise -> error ("Given shape: " ++ show s2 ++ " has unit dimensions that would need to be broadcasted to input tensor shape")

sliceUsafe :: Tensor s a -> [Slice] -> Tensor s a
sliceUsafe (Tensor {shape = s, array = a}) slices =
  let normSlices = normalizeSlices s slices
      indexIntervals = zipWith toIndex s normSlices
      newShape = map length indexIntervals
      newStrides = stridesFromShape newShape
      origStrides = stridesFromShape s
      totalElements = product newShape
      newArray =
        V.generate totalElements $ \i ->
          let newIndices = posToIndices newStrides i
              oldIndices = zipWith (\idx range -> idx + (range |> listToMaybe |> fromJust)) newIndices indexIntervals
           in indexFromStride a origStrides oldIndices
   in Tensor {shape = newShape, array = newArray}
  where
    toIndex _ (Range start end) = [start .. end - 1]
    toIndex _ (Single idx) = [idx]
    toIndex s All = [0 .. s - 1]

slice :: forall s a n. (Length s ~ n) => Tensor s a -> LList n Slice -> Tensor s a
slice t s = sliceUsafe t (toList s)
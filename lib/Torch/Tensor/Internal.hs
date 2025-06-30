module Torch.Tensor.Internal where

import Control.DeepSeq (NFData, rnf)
import Data.List (intercalate)
import Data.Vector (Vector, (!))
import Data.Vector qualified as V
import Debug.Trace (traceShow)
import GHC.TypeLits (natVal)
import Torch.Tensor.Types
import Torch.Utils

toList :: LList n a -> [a]
toList LNil = []
toList (x :~ xs) = x : toList xs

toIntList :: IList l -> [Int]
toIntList INil = []
toIntList (n :- xs) = fromIntegral (natVal n) : toIntList xs

toInt :: Fin n -> Int
toInt FinZ = 0
toInt (FinS n) = 1 + toInt n

validateShape :: [Int] -> Either String ()
validateShape [] = Left "Shape cannot be empty"
validateShape shape
  | any (< 0) shape = Left "Shape cannot contain negative dimensions"
  | otherwise = Right ()

normalizeIndices :: [Int] -> [Int] -> [Int]
normalizeIndices shape indices =
  let normalized = zipWith (\s i -> if i < 0 then s + i else i) shape indices
   in if any (< 0) normalized || or (zipWith (<=) shape normalized)
        then error "Indices out of bounds"
        else normalized

normalizeAxes :: Int -> [Int] -> [Int]
normalizeAxes rank axes =
  let normalized = map (\a -> if a < 0 then rank + a else a) axes
   in if any (< 0) normalized || any (>= rank) normalized
        then error "Axes out of bounds"
        else normalized

normalizeSlices :: [Int] -> [Slice] -> [Slice]
normalizeSlices shape slices =
  zipWith normalizeSlice shape slices
  where
    normalizeSlice :: Int -> Slice -> Slice
    normalizeSlice _ All = All
    normalizeSlice s (Range start end) =
      let start' = if start < 0 then s + start else start
          end' = if end < 0 then s + end else end
       in if start' < 0 || end' > s || start' >= end'
            then error "Invalid range"
            else Range start' end'
    normalizeSlice s (Single idx) =
      let idx' = if idx < 0 then s + idx else idx
       in if idx' < 0 || idx' >= s
            then error "Index out of bounds"
            else Single idx'

normalizeReshape :: [Int] -> [Int] -> [Int]
normalizeReshape oldShape newShape
  | count (-1) newShape > 1 =
      error "Cannot reshape tensor with multiple -1 dimensions"
  | count (-1) newShape == 1 =
      let knownSize = product $ filter (/= -1) newShape
          totalSize = product oldShape
       in if totalSize `mod` knownSize /= 0
            then error "Cannot reshape tensor with -1 dimension to fit the total size"
            else
              let missingSize = totalSize `div` knownSize
               in map (\x -> if x == -1 then missingSize else x) newShape
  | otherwise =
      if product newShape /= product oldShape
        then error "Cannot reshape tensor to a different size"
        else newShape
  where
    count x = length . filter (== x)

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

broadcastArray :: Vector a -> [Int] -> [Int] -> Vector a
broadcastArray arr oldShape newShape
  | oldShape == newShape = arr
  | otherwise =
      let broadcastStrides = stridesForBroadcast oldShape newShape
          totalElements = product newShape
          newArray = V.generate totalElements $ \i ->
            let indices = posToIndices (stridesFromShape newShape) i
             in indexFromStride arr broadcastStrides indices
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

unsqueezeShapeToIfPossible :: [Int] -> [Int] -> [Int]
unsqueezeShapeToIfPossible oldShape newShape
  | length oldShape >= length newShape = oldShape
  | otherwise =
      replicate (length newShape - length oldShape) 1 ++ oldShape

instance (NFData a) => NFData (Tensor s a) where
  rnf (Tensor shape array) = rnf (shape, array)

instance (Show a) => Show (Tensor s a) where
  show (Tensor sh arr) =
    "Tensor " ++ show sh ++ ":\n" ++ formatArray sh (V.toList arr)

formatArray :: (Show a) => [Int] -> [a] -> String
formatArray [] xs = error "Tensor has no dimensions"
formatArray [d] xs = "[" ++ intercalate ", " (map show xs) ++ "]"
formatArray (d : ds) xs =
  let chunked = chunksOf (product ds) xs
   in "[" ++ intercalate ",\n " (map (formatArray ds) chunked) ++ "]"
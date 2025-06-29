module Tensor.Internal where

import Data.List (intercalate)
import Data.Vector (Vector, (!))
import Data.Vector qualified as V
import Tensor.Types

toList :: LList n a -> [a]
toList LNil = []
toList (x :~ xs) = x : toList xs

toInt :: Fin n -> Int
toInt FinZ = 0
toInt (FinS n) = 1 + toInt n

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

instance (Show a) => Show (Tensor s a) where
  show (Tensor sh arr) =
    "Tensor " ++ show sh ++ ":\n" ++ formatArray sh (V.toList arr)

formatArray :: (Show a) => [Int] -> [a] -> String
formatArray [] xs = error "Tensor has no dimensions"
formatArray [d] xs = "[" ++ intercalate ", " (map show xs) ++ "]"
formatArray (d : ds) xs =
  let chunked = chunksOf (product ds) xs
   in "[" ++ intercalate ",\n " (map (formatArray ds) chunked) ++ "]"

chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs = let (h, t) = splitAt n xs in h : chunksOf n t
module Torch.Utils where

(|>) :: a -> (a -> b) -> b
x |> f = f x

class BoolKindToValue (b :: Bool) where
  toValue :: Bool

instance BoolKindToValue 'True where
  toValue = True

instance BoolKindToValue 'False where
  toValue = False

chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs = let (h, t) = splitAt n xs in h : chunksOf n t

swap :: [Int] -> Int -> Int -> [Int]
swap xs i j
  | i < 0 || j < 0 || i >= length xs || j >= length xs = error "Index out of bounds"
  | otherwise = [choose k | k <- [0 .. length xs - 1]]
  where
    choose :: Int -> Int
    choose k
      | k == i = xs !! j
      | k == j = xs !! i
      | otherwise = xs !! k
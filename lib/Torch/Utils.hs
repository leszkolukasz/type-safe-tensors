module Torch.Utils where

(|>) :: a -> (a -> b) -> b
x |> f = f x

class BoolKindToValue (b :: Bool) where
  toValue :: Bool

instance BoolKindToValue 'True where
  toValue = True

instance BoolKindToValue 'False where
  toValue = False

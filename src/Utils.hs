module Utils where

(|>) :: a -> (a -> b) -> b
x |> f = f x

class ShowBoolKind (b :: Bool) where
  showBool :: String

instance ShowBoolKind 'True where
  showBool = "True"

instance ShowBoolKind 'False where
  showBool = "False"

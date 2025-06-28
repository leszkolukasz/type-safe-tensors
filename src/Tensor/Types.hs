module Tensor.Types where

import Data.Kind (Type)
import Data.Vector (Vector)
import Data.Vector qualified as V
import GHC.TypeLits (Nat, Symbol, type (+))

type Any = "Any"

type Shape = [Symbol]

data Tensor (s :: Shape) (a :: Type) where
  Tensor :: {shape :: [Int], array :: Vector a} -> Tensor s a
  deriving (Show)

type DoubleTensor s = Tensor s Double

type FloatTensor s = Tensor s Float

type IntTensor s = Tensor s Int

type family Compatible (s1 :: Shape) (s2 :: Shape) :: Bool where
  Compatible '[] '[] = 'True
  Compatible (n : s1) (n : s2) = Compatible s1 s2
  Compatible (Any : s1) (_ : s2) = Compatible s1 s2
  Compatible (_ : s1) (Any : s2) = Compatible s1 s2
  Compatible _ _ = 'False

infixr 5 :~

data LList (n :: Nat) (a :: Type) where
  LNil :: LList 0 a
  (:~) :: a -> LList n a -> LList (n + 1) a

type family Length (l :: [a]) :: Nat where
  Length '[] = 0
  Length (_ : xs) = 1 + Length xs
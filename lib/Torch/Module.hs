module Torch.Module where

import Data.Kind (Type)
import GHC.TypeLits (Symbol)
import Torch.Tensor
import Torch.Tensor.Types

data Linear (inFeatures :: Symbol) (outFeatures :: Symbol) (a :: Type) where
  Linear :: {weight :: Tensor '[outFeatures, inFeatures] a, bias :: Maybe (Tensor '[outFeatures] a)} -> Linear inFeatures outFeatures a

linear ::
  forall inFeatures outFeatures s1 s2 n1 n2 m1 m2 a.
  ( s1 ~ '[outFeatures, inFeatures],
    Broadcastable (DropLastN s1 2) (DropLastN s2 2) ~ 'True,
    SndToLast s1 ~ 'Just n1,
    Last s1 ~ 'Just n2,
    SndToLast s2 ~ 'Just m1,
    Last s2 ~ 'Just m2,
    Compatible '[n2] '[m1] ~ 'True,
    Num a
  ) =>
  Linear inFeatures outFeatures a ->
  Tensor s2 a ->
  Tensor
    ( Concat
        ( BroadcastUnion
            (DropLastN s1 2)
            (DropLastN s2 2)
        )
        [outFeatures, m2]
    )
    a
linear (Linear {weight = weight, bias = bias}) input =
  let result = weight @. input
   in result
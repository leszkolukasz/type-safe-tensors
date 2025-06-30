module Torch.Module
  ( Linear (..),
    linear,
    relu,
    softmax,
  )
where

import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import Data.Vector qualified as V
import GHC.TypeLits (Symbol)
import Torch.Tensor
import Torch.Tensor.Types

data Linear (inFeatures :: Symbol) (outFeatures :: Symbol) (a :: Type) where
  Linear :: {weight :: Tensor '[inFeatures, outFeatures] a, bias :: Tensor '[outFeatures] a} -> Linear inFeatures outFeatures a

linear ::
  forall inFeatures outFeatures s1 s2 n1 n2 m1 m2 a.
  ( s2 ~ '[inFeatures, outFeatures],
    Broadcastable (DropLastN s1 2) (DropLastN s2 2) ~ 'True,
    SndToLast s1 ~ 'Just n1,
    Last s1 ~ 'Just n2,
    SndToLast s2 ~ 'Just m1,
    Last s2 ~ 'Just m2,
    Compatible '[n2] '[m1] ~ 'True,
    Broadcastable (Concat (DropLastN s1 2) [n1, outFeatures]) '[outFeatures] ~ 'True,
    Num a
  ) =>
  Linear inFeatures outFeatures a ->
  Tensor s1 a ->
  Tensor (BroadcastUnion (Concat (DropLastN s1 2) [n1, m2]) '[outFeatures]) a
linear (Linear {weight = weight, bias = bias}) input =
  let x :: Tensor (Concat (DropLastN s1 2) [n1, outFeatures]) a = input @. weight
   in x +. bias

relu :: (Ord a, Num a) => Tensor s a -> Tensor s a
relu = fmap (`max` 0)

softmax :: forall a s1 s2. (Ord a, Floating a) => Tensor '[s1, s2] a -> Tensor '[s1, s2] a
softmax t =
  let maxVal :: Tensor '[s1] a = reduce V.maximum t (Proxy @1 :- INil)
      maxVal' :: Tensor '[s1, Any] a = unsqueeze @1 maxVal
      expT :: Tensor '[s1, s2] a = fmap exp (t -. maxVal')
      sumExp :: Tensor '[s1] a = reduce V.sum expT (Proxy @1 :- INil)
      sumExp' :: Tensor '[s1, Any] a = unsqueeze @1 sumExp
   in expT /. sumExp'
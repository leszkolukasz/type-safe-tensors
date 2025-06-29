module Module where

import Data.Kind (Type)
import GHC.TypeLits (Symbol)
import Tensor.Types

-- class Module (m: Shape -> Shape) where
--   forward :: Tensor m inp a -> Tensor m out a

-- class Module (inp :: Shape) (out :: Shape) where
--   forward :: Tensor inp a -> Tensor out a

-- newtype Linear (inpD :: Symbol) (outD :: Symbol) (a :: Type) = Linear
--   {weights :: Tensor '[inpD, outD] a}

-- instance (Num a) => Module '[Any, inpD] '[Any, outD] where
--   forward t1 = undefined
module Tensor
  ( Tensor (..),
    DoubleTensor,
    FloatTensor,
    IntTensor,
    LList (..),
    Compatible,
    fromList,
    (+.),
    (-.),
    (*.),
    (/.),
    indexUnsafe,
    index,
    setUnsafe,
    set,
  )
where

import Tensor.Internal
import Tensor.Op
import Tensor.Types

module TypeSpec where

import Data.Type.Equality
import Torch.Tensor.Types

test :: a :~: b -> ()
test proof =
  case proof of
    Refl -> ()

-- Compatible
compatible1 :: ()
compatible1 = test (Refl :: Compatible '[] '[] :~: 'True)

compatible2 :: ()
compatible2 = test (Refl :: Compatible '["a"] '[Any] :~: 'True)

compatible3 :: ()
compatible3 = test (Refl :: Compatible '["a"] '[Any, "c"] :~: 'False)

compatible4 :: ()
compatible4 = test (Refl :: Compatible '[Any, "b"] '["c"] :~: 'False)

compatible5 :: ()
compatible5 = test (Refl :: Compatible '["a"] '["b", "c"] :~: 'False)

compatible6 :: ()
compatible6 = test (Refl :: Compatible '["a", "b"] '["a", "b"] :~: 'True)

compatible7 :: ()
compatible7 = test (Refl :: Compatible '["b", "a"] '["a", "b"] :~: 'False)

-- -- Broadcastable
broadcastable1 :: ()
broadcastable1 = test (Refl :: Broadcastable '[] '[] :~: 'True)

broadcastable2 :: ()
broadcastable2 = test (Refl :: Broadcastable '["a"] '[] :~: 'True)

broadcastable3 :: ()
broadcastable3 = test (Refl :: Broadcastable '[] '["b"] :~: 'True)

broadcastable4 :: ()
broadcastable4 = test (Refl :: Broadcastable '["a", Any] '["a", "b"] :~: 'True)

broadcastable5 :: ()
broadcastable5 = test (Refl :: Broadcastable '["a", "b"] '["c", "b"] :~: 'False)

broadcastable6 :: ()
broadcastable6 = test (Refl :: Broadcastable '["a", "b", "c"] '[] :~: 'True)

broadcastable7 :: ()
broadcastable7 = test (Refl :: Broadcastable '[Any] '["a", "b", "c"] :~: 'True)

broadcastable8 :: ()
broadcastable8 = test (Refl :: Broadcastable '["a"] '["a", "b", "c"] :~: 'False)

-- Union
union1 :: ()
union1 = test (Refl :: Union '[] '[] :~: '[])

union2 :: ()
union2 = test (Refl :: Union '["a"] '["a"] :~: '["a"])

union3 :: ()
union3 = test (Refl :: Union '[Any, "b"] '["c", "b"] :~: '["c", "b"])

union4 :: ()
union4 = test (Refl :: Union '["a", Any] '["a", "d"] :~: '["a", "d"])

-- BroadcastUnion
broadcastUnion1 :: ()
broadcastUnion1 = test (Refl :: BroadcastUnion '[] '[] :~: '[])

broadcastUnion2 :: ()
broadcastUnion2 = test (Refl :: BroadcastUnion '["a"] '[] :~: '["a"])

broadcastUnion3 :: ()
broadcastUnion3 = test (Refl :: BroadcastUnion '[] '["b"] :~: '["b"])

broadcastUnion4 :: ()
broadcastUnion4 = test (Refl :: BroadcastUnion '["a", Any] '["a", "b"] :~: '["a", "b"])

broadcastUnion5 :: ()
broadcastUnion5 = test (Refl :: BroadcastUnion '["c"] '["a", "b", "c"] :~: '["a", "b", "c"])

-- Equal
equal1 :: ()
equal1 = test (Refl :: Equal '[] '[] :~: 'True)

equal2 :: ()
equal2 = test (Refl :: Equal '["a"] '["a"] :~: 'True)

equal3 :: ()
equal3 = test (Refl :: Equal '[Any] '["a"] :~: 'False)

equal4 :: ()
equal4 = test (Refl :: Equal '[Any] '[] :~: 'False)

equal5 :: ()
equal5 = test (Refl :: Equal '["a"] '["b"] :~: 'False)

equal6 :: ()
equal6 = test (Refl :: Equal '["a", "b"] '["a", "b"] :~: 'True)

equal7 :: ()
equal7 = test (Refl :: Equal '["a", "b"] '["b", "a"] :~: 'False)

-- FirstTuple
firstTuple1 :: ()
firstTuple1 = test (Refl :: FirstTuple '(a, b) :~: a)

-- SecondTuple
secondTuple1 :: ()
secondTuple1 = test (Refl :: SecondTuple '(a, b) :~: b)

-- Last
last1 :: ()
last1 = test (Refl :: Last '[] :~: 'Nothing)

last2 :: ()
last2 = test (Refl :: Last '[a] :~: 'Just a)

last3 :: ()
last3 = test (Refl :: Last '[a, b] :~: 'Just b)

-- SndToLast
sndToLast1 :: ()
sndToLast1 = test (Refl :: SndToLast '[] :~: 'Nothing)

sndToLast2 :: ()
sndToLast2 = test (Refl :: SndToLast '[a] :~: 'Nothing)

sndToLast3 :: ()
sndToLast3 = test (Refl :: SndToLast '[a, b] :~: 'Just a)

sndToLast4 :: ()
sndToLast4 = test (Refl :: SndToLast '[a, b, c] :~: 'Just b)

-- Length
length1 :: ()
length1 = test (Refl :: Length '[] :~: 0)

length2 :: ()
length2 = test (Refl :: Length '[a] :~: 1)

length3 :: ()
length3 = test (Refl :: Length '[a, b, c] :~: 3)

-- Concat
concat1 :: ()
concat1 = test (Refl :: Concat '[] '[] :~: '[])

concat2 :: ()
concat2 = test (Refl :: Concat '[a] '[b] :~: '[a, b])

concat3 :: ()
concat3 = test (Refl :: Concat '[a, b] '[c, d] :~: '[a, b, c, d])

-- Reverse
reverse1 :: ()
reverse1 = test (Refl :: Reverse '[] :~: '[])

reverse2 :: ()
reverse2 = test (Refl :: Reverse '[a] :~: '[a])

reverse3 :: ()
reverse3 = test (Refl :: Reverse '[a, b, c] :~: '[c, b, a])

-- DropFirstN
dropFirstN1 :: ()
dropFirstN1 = test (Refl :: DropFirstN '[] 2 :~: '[])

dropFirstN2 :: ()
dropFirstN2 = test (Refl :: DropFirstN '[a, b, c] 0 :~: '[a, b, c])

dropFirstN3 :: ()
dropFirstN3 = test (Refl :: DropFirstN '[a, b, c] 2 :~: '[c])

-- DropLastN
dropLastN1 :: ()
dropLastN1 = test (Refl :: DropLastN '[] 2 :~: '[])

dropLastN2 :: ()
dropLastN2 = test (Refl :: DropLastN '[a, b, c] 0 :~: '[a, b, c])

dropLastN3 :: ()
dropLastN3 = test (Refl :: DropLastN '[a, b, c] 2 :~: '[a])

-- TakeFirstN
takeFirstN1 :: ()
takeFirstN1 = test (Refl :: TakeFirstN '[a, b, c] 2 :~: '[a, b])

takeFirstN2 :: ()
takeFirstN2 = test (Refl :: TakeFirstN '[a, b, c] 0 :~: '[])

-- TakeLastN
takeLastN1 :: ()
takeLastN1 = test (Refl :: TakeLastN '[a, b, c] 2 :~: '[b, c])

takeLastN2 :: ()
takeLastN2 = test (Refl :: TakeLastN '[a, b, c] 0 :~: '[])

-- Split
split1 :: ()
split1 = test (Refl :: Split '[a, b, c] 2 :~: '( '[a, b], '[c]))

split2 :: ()
split2 = test (Refl :: Split '[a, b, c] 0 :~: '( '[], '[a, b, c]))

-- DropNth
dropNth1 :: ()
dropNth1 = test (Refl :: DropNth '[] 2 :~: '[])

dropNth2 :: ()
dropNth2 = test (Refl :: DropNth '[a, b, c] 0 :~: '[b, c])

dropNth3 :: ()
dropNth3 = test (Refl :: DropNth '[a, b, c] 2 :~: '[a, b])

-- Unsqueeze
unsqueeze1 :: ()
unsqueeze1 = test (Refl :: Unsqueeze '[a, b, c] 1 :~: '[a, Any, b, c])

unsqueeze2 :: ()
unsqueeze2 = test (Refl :: Unsqueeze '[a, b, c] 0 :~: '[Any, a, b, c])

-- Prepend
prepend1 :: ()
prepend1 = test (Refl :: Prepend '[a, b] 0 c :~: '[a, b])

prepend2 :: ()
prepend2 = test (Refl :: Prepend '[a, b] 2 c :~: '[c, c, a, b])

-- Nth
nth1 :: ()
nth1 = test (Refl :: Nth '[a, b, c] 0 :~: a)

nth2 :: ()
nth2 = test (Refl :: Nth '[a, b, c] 1 :~: b)

nth3 :: ()
nth3 = test (Refl :: Nth '[a, b, c] 2 :~: c)

-- Swap
swap1 :: ()
swap1 = test (Refl :: Swap '[a, b, c] 0 1 :~: '[b, a, c])

swap2 :: ()
swap2 = test (Refl :: Swap '[a, b, c] 1 2 :~: '[a, c, b])

swap3 :: ()
swap3 = test (Refl :: Swap '[a, b, c] 0 2 :~: '[c, b, a])

swap4 :: ()
swap4 = test (Refl :: Swap '[a, b, c] 1 1 :~: '[a, b, c])

-- LtNat
ltNat1 :: ()
ltNat1 = test (Refl :: LtNat 1 2 :~: 'True)

ltNat2 :: ()
ltNat2 = test (Refl :: LtNat 2 1 :~: 'False)

ltNat3 :: ()
ltNat3 = test (Refl :: LtNat 2 2 :~: 'False)

-- EqNat

eqNat1 :: ()
eqNat1 = test (Refl :: EqNat 1 1 :~: 'True)

eqNat2 :: ()
eqNat2 = test (Refl :: EqNat 1 2 :~: 'False)

-- AllBelow

allBelow1 :: ()
allBelow1 = test (Refl :: AllBelow '[] 0 :~: 'True)

allBelow2 :: ()
allBelow2 = test (Refl :: AllBelow '[1, 2, 3] 4 :~: 'True)

allBelow3 :: ()
allBelow3 = test (Refl :: AllBelow '[1, 2, 3] 3 :~: 'False)

allBelow4 :: ()
allBelow4 = test (Refl :: AllBelow '[1, 2, 3] 2 :~: 'False)

-- Elem
elem1 :: ()
elem1 = test (Refl :: Elem 1 '[1, 2, 3] :~: 'True)

elem2 :: ()
elem2 = test (Refl :: Elem 4 '[1, 2, 3] :~: 'False)

-- Range
range1 :: ()
range1 = test (Refl :: Range 0 :~: '[])

range2 :: ()
range2 = test (Refl :: Range 3 :~: '[0, 1, 2])

-- RemoveDims

removeDims1 :: ()
removeDims1 = test (Refl :: RemoveDims '[] '[] Any :~: '[Any])

removeDims2 :: ()
removeDims2 = test (Refl :: RemoveDims '[a, b, c] '[] Any :~: '[a, b, c])

removeDims3 :: ()
removeDims3 = test (Refl :: RemoveDims '[a, b, c] '[0] Any :~: '[b, c])

removeDims4 :: ()
removeDims4 = test (Refl :: RemoveDims '[a, b, c] '[1] Any :~: '[a, c])

removeDims5 :: ()
removeDims5 = test (Refl :: RemoveDims '[a, b, c] '[0, 2] Any :~: '[b])

removeDims6 :: ()
removeDims6 = test (Refl :: RemoveDims '[a, b, c] '[2, 1, 0] Any :~: '[Any])

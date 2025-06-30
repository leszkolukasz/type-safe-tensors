module Torch.Tensor.Types where

import Data.Data (Proxy (..))
import Data.Kind (Type)
import Data.Vector (Vector)
import Data.Vector qualified as V
import GHC.TypeLits (CmpNat, ErrorMessage (..), KnownNat, Nat, OrderingI (..), Symbol, TypeError, type (+), type (-), type (<=))

type Any = "Any"

type Shape = [Symbol]

data Tensor (s :: Shape) (a :: Type) where
  Tensor :: {shape :: [Int], array :: Vector a} -> Tensor s a
  deriving (Eq)

type DoubleTensor s = Tensor s Double

type FloatTensor s = Tensor s Float

type IntTensor s = Tensor s Int

type family Compatible (s1 :: Shape) (s2 :: Shape) :: Bool where
  Compatible '[] '[] = 'True
  Compatible (n : s1) (n : s2) = Compatible s1 s2
  Compatible (Any : s1) (_ : s2) = Compatible s1 s2
  Compatible (_ : s1) (Any : s2) = Compatible s1 s2
  Compatible _ _ = 'False

type family Broadcastable (s1 :: Shape) (s2 :: Shape) :: Bool where
  Broadcastable l1 l2 = BroadcastableGo (Reverse l1) (Reverse l2)

type family BroadcastableGo (s1 :: [Symbol]) (s2 :: [Symbol]) :: Bool where
  BroadcastableGo '[] _ = 'True
  BroadcastableGo _ '[] = 'True
  BroadcastableGo (Any : s1) (_ : s2) = BroadcastableGo s1 s2
  BroadcastableGo (_ : s1) (Any : s2) = BroadcastableGo s1 s2
  BroadcastableGo (n : s1) (n : s2) = BroadcastableGo s1 s2
  BroadcastableGo _ _ = 'False

type family Union (s1 :: Shape) (s2 :: Shape) :: Shape where
  Union '[] '[] = '[]
  Union (n : s1) (n : s2) = n : Union s1 s2
  Union (Any : s1) (n : s2) = n : Union s1 s2
  Union (n : s1) (Any : s2) = n : Union s1 s2
  Union _ _ = TypeError ('Text "Cannot unionize incompatible shapes")

type family BroadcastUnion (s1 :: Shape) (s2 :: Shape) :: Shape where
  BroadcastUnion s1 s2 = Reverse (BroadcastUnionGo (Reverse s1) (Reverse s2))

type family BroadcastUnionGo (s1 :: [Symbol]) (s2 :: [Symbol]) :: Shape where
  BroadcastUnionGo '[] '[] = '[]
  BroadcastUnionGo '[] (n : s2) = n : BroadcastUnionGo '[] s2
  BroadcastUnionGo (n : s1) '[] = n : BroadcastUnionGo s1 '[]
  BroadcastUnionGo (Any : s1) (n : s2) = n : BroadcastUnionGo s1 s2
  BroadcastUnionGo (n : s1) (Any : s2) = n : BroadcastUnionGo s1 s2
  BroadcastUnionGo (n : s1) (n : s2) = n : BroadcastUnionGo s1 s2
  BroadcastUnionGo _ _ = TypeError ('Text "Cannot broadcast-unionize incompatible shapes")

type family Equal (s1 :: Shape) (s2 :: Shape) :: Bool where
  Equal '[] '[] = 'True
  Equal (n : s1) (n : s2) = Equal s1 s2
  Equal _ _ = 'False

type family FirstTuple (l :: (a, b)) :: a where
  FirstTuple '(x, _) = x

type family SecondTuple (l :: (a, b)) :: b where
  SecondTuple '(_, y) = y

type family Last (l :: [a]) :: Maybe a where
  Last '[] = 'Nothing
  Last '[x] = 'Just x
  Last (_ : xs) = Last xs

type family SndToLast (l :: [a]) :: Maybe a where
  SndToLast '[] = 'Nothing
  SndToLast '[x] = 'Nothing
  SndToLast '[y, _] = 'Just y
  SndToLast (_ : xs) = SndToLast xs

type family Length (l :: [a]) :: Nat where
  Length '[] = 0
  Length (_ : xs) = 1 + Length xs

type family Concat (l1 :: [a]) (l2 :: [a]) :: [a] where
  Concat '[] l2 = l2
  Concat (x : xs) l2 = x : Concat xs l2

type family Reverse (l :: [a]) :: [a] where
  Reverse '[] = '[]
  Reverse (x : xs) = Concat (Reverse xs) '[x]

type family DropFirstN (l :: [a]) (n :: Nat) :: [a] where
  DropFirstN '[] _ = '[]
  DropFirstN xs 0 = xs
  DropFirstN (_ : xs) n = DropFirstN xs (n - 1)

type family DropLastN (l :: [a]) (n :: Nat) :: [a] where
  DropLastN '[] _ = '[]
  DropLastN xs 0 = xs
  DropLastN xs n = Reverse (DropFirstN (Reverse xs) n)

type family TakeFirstN (l :: [a]) (n :: Nat) :: [a] where
  TakeFirstN l n = DropLastN l (Length l - n)

type family TakeLastN (l :: [a]) (n :: Nat) :: [a] where
  TakeLastN l n = DropFirstN l (Length l - n)

type family Split (l :: [a]) (n :: Nat) :: ([a], [a]) where
  Split l n = '(TakeFirstN l n, DropFirstN l n)

type family DropNth (l :: [a]) (n :: Nat) :: [a] where
  DropNth '[] _ = '[]
  DropNth (x : xs) 0 = xs
  DropNth (x : xs) n = x : DropNth xs (n - 1)

type family Unsqueeze (s :: Shape) (n :: Nat) :: Shape where
  Unsqueeze s n = Concat (TakeFirstN s n) (Any : DropFirstN s n)

type family Prepend (s :: Shape) (n :: Nat) (v :: Symbol) :: Shape where
  Prepend s 0 v = s
  Prepend l n v = v : Prepend l (n - 1) v

type family Nth (l :: [a]) (n :: Nat) :: a where
  Nth '[] _ = TypeError ('Text "Index out of bounds")
  Nth (x : xs) 0 = x
  Nth (_ : xs) n = Nth xs (n - 1)

type family Swap (l :: [a]) (i :: Nat) (j :: Nat) :: [a] where
  Swap l i j = SwapGo l l i j 0

type family SwapGo (l :: [a]) (orig :: [a]) (i :: Nat) (j :: Nat) (idx :: Nat) :: [a] where
  SwapGo '[] _ _ _ _ = '[]
  SwapGo (x : xs) orig i j i = Nth orig j : SwapGo xs orig i j (i + 1)
  SwapGo (x : xs) orig i j j = Nth orig i : SwapGo xs orig i j (j + 1)
  SwapGo (x : xs) orig i j idx = x : SwapGo xs orig i j (idx + 1)

type family Not (b :: Bool) :: Bool where
  Not 'True = 'False
  Not 'False = 'True

type family LtNat (x :: Nat) (y :: Nat) :: Bool where
  LtNat x y = LtNatGo (CmpNat x y)

type family LtNatGo (x :: Ordering) :: Bool where
  LtNatGo 'LT = 'True
  LtNatGo 'EQ = 'False
  LtNatGo 'GT = 'False

type family EqNat (x :: Nat) (y :: Nat) :: Bool where
  EqNat x y = EqNatGo (CmpNat x y)

type family EqNatGo (x :: Ordering) :: Bool where
  EqNatGo 'LT = 'False
  EqNatGo 'EQ = 'True
  EqNatGo 'GT = 'False

type family And (a :: Bool) (b :: Bool) :: Bool where
  And 'True 'True = 'True
  And _ _ = 'False

type family Or (a :: Bool) (b :: Bool) :: Bool where
  Or 'False 'False = 'False
  Or _ _ = 'True

type family AllBelow (l :: [Nat]) (n :: Nat) :: Bool where
  AllBelow '[] _ = 'True
  AllBelow (x : xs) n = And (LtNat x n) (AllBelow xs n)

type family Elem (x :: Nat) (l :: [Nat]) :: Bool where
  Elem _ '[] = 'False
  Elem x (y : ys) = Or (EqNat x y) (Elem x ys)

type family Range (end :: Nat) :: [Nat] where
  Range n = RangeGo n 0

type family RangeGo (end :: Nat) (idx :: Nat) :: [Nat] where
  RangeGo end end = '[]
  RangeGo end idx = idx : RangeGo end (idx + 1)

type family RemoveDims (s :: [a]) (l :: [Nat]) :: [a] where
  RemoveDims s l = RemoveDimsGo s l (Range (Length s))

type family RemoveDimsGo (s :: [a]) (l :: [Nat]) (idxs :: [Nat]) :: [a] where
  RemoveDimsGo '[] _ _ = '[]
  RemoveDimsGo _ _ '[] = '[]
  RemoveDimsGo (n : s) l (i : idxs) = Concat (RemoveDimsGo2 (n : s) (Elem i l)) (RemoveDimsGo s l idxs)

type family RemoveDimsGo2 (s :: [a]) (e :: Bool) :: [a] where
  RemoveDimsGo2 '[] _ = '[]
  RemoveDimsGo2 (n : s) 'True = '[]
  RemoveDimsGo2 (n : s) 'False = '[n]

type Reductor a = Vector a -> a

infixr 5 :~

infixr 5 :-

data LList (n :: Nat) (a :: Type) where
  LNil :: LList 0 a
  (:~) :: a -> LList n a -> LList (n + 1) a

data IList (l :: [Nat]) where
  INil :: IList '[]
  (:-) :: forall n l. (KnownNat n) => Proxy n -> IList l -> IList (n ': l)

data Fin n where
  FinZ :: Fin (n + 1)
  FinS :: Fin n -> Fin (n + 1)

data Slice
  = Range Int Int
  | Single Int
  | All
  deriving (Show)
module Playground where
    import Data.Kind
    import Control.Exception (TypeError)
    import GHC.TypeError
    
    data Expr a where
        I :: Int -> Expr Int
        B :: Bool -> Expr Bool
        Add :: Expr Int -> Expr Int -> Expr Int
        Eq  :: Expr Int -> Expr Int -> Expr Bool

    eval :: Expr a -> a
    eval (I n)       = n
    eval (B b)       = b
    eval (Add e1 e2) = eval e1 + eval e2
    eval (Eq  e1 e2) = eval e1 == eval e2


    data Nat :: Type where
        Z :: Nat
        S :: Nat -> Nat

    data HList :: [Type] -> Type where
        HNil  :: HList '[]
        HCons :: a -> HList t -> HList (getType a ': t)

    data Tuple :: (Type,Type) -> Type where
        Tuple :: a -> b -> Tuple '(a,b)

    data Container :: Type where
        Container :: a -> Container


    type family Add (n :: Nat) (m :: Nat) :: Nat
    type instance Add 'Z m = m
    type instance Add ('S n) m = 'S (Add n m)

    infixr 6 :>
    data Vec :: Nat -> Type -> Type where
        V0   :: Vec 'Z a
        (:>) :: a -> Vec n a -> Vec ('S n) a

    deriving instance (Show a) => Show (Vec n a)

    vhead :: Vec ('S n) a -> a
    vhead (x:>_) = x

    vtail :: Vec ('S n) a -> Vec n a
    vtail (_:> xs) = xs

    vapp :: Vec m a -> Vec n a -> Vec (Add m n) a
    vapp V0 ys = ys
    vapp (x:>xs) ys = x:>(vapp xs ys)

    type SNat :: Nat -> Type
    data SNat n where
        SZ :: SNat 'Z
        SS :: SNat n -> SNat ('S n)

    deriving instance Show(SNat n)

    vreplicate :: SNat n -> a -> Vec n a
    vreplicate SZ _ = V0
    vreplicate (SS n) x = x:>(vreplicate n x)
    
    data Proxy :: k -> Type where
        Proxy :: Proxy (i::k)

    vtake3 :: SNat m -> Proxy n -> Vec (Add m n) a -> Vec m a
    vtake3 SZ     _ _ = V0
    vtake3 (SS m) n (x:>xs) = x :> vtake3 m n xs

    type family Head (a :: [k]) :: k where
        Head '[] = GHC.TypeError.TypeError ('Text "Empty list")
        Head (x:_) = x

    -- data Test :: Type -> Type where
    --     Test :: Test 'Z

    data Test :: Nat -> Type where
        Test :: Test 'Z
module Torch.Tensor.Template where

import Control.Monad (when)
import Data.Foldable (foldrM)
import Data.Vector qualified as V
import GHC.Base (build)
import Language.Haskell.TH
import Language.Haskell.TH.Syntax (mkName)
import Torch.Tensor.Types

buildNestedNArrayType :: (Quote m) => Int -> m (Type, Name)
buildNestedNArrayType 1 = do
  a <- newName "a"
  arrTyp <- [t|[$(varT a)]|]
  return (arrTyp, a)
buildNestedNArrayType n = do
  (innerTyp, a) <- buildNestedNArrayType (n - 1)
  arrTyp <- [t|[$(return innerTyp)]|]
  return (arrTyp, a)

buildTensorShapeType :: (Quote m) => Int -> m Type
buildTensorShapeType 0 = [t|'[]|]
buildTensorShapeType n = do
  s <- newName "s"
  rest <- buildTensorShapeType (n - 1)
  [t|$(varT s) ': $(return rest)|]

buildValidateN :: (Quote m) => Int -> m Dec
buildValidateN 1 = do
  let name = mkName "validate1"
  l <- newName "l"
  let params = [VarP l]
  body <-
    [|
      do
        when (null $(varE l)) $ Left "Input cannot be empty"
        Right [length $(varE l)]
      |]
  return $ FunD name [Clause params (NormalB body) []]
buildValidateN n = do
  let name = mkName $ "validate" ++ show n
  let prevName = mkName $ "validate" ++ show (n - 1)
  l <- newName "l"
  let params = [VarP l]
  body <-
    [|
      do
        when (null $(varE l)) $ Left "Input cannot be empty"
        nested <- mapM $(varE prevName) $(varE l)
        when (any (/= head nested) nested) $ Left "All inner lists must have the same length"
        Right $ length $(varE l) : head nested
      |]
  return $ FunD name [Clause params (NormalB body) []]

buildValidateNs :: (Quote m) => Int -> m [Dec]
buildValidateNs n = mapM buildValidateN [1 .. n]

buildFlattenN :: (Quote m) => Int -> m Dec
buildFlattenN 1 = do
  let name = mkName "flatten1"
  l <- newName "l"
  let params = [VarP l]
  body <- [|concat $(varE l)|]
  return $ FunD name [Clause params (NormalB body) []]
buildFlattenN n = do
  let name = mkName $ "flatten" ++ show n
  let prevName = mkName $ "flatten" ++ show (n - 1)
  l <- newName "l"
  let params = [VarP l]
  body <- [|concatMap $(varE prevName) $(varE l)|]
  return $ FunD name [Clause params (NormalB body) []]

buildFlattenNs :: (Quote m) => Int -> m [Dec]
buildFlattenNs n = mapM buildFlattenN [1 .. n]

buildFromNestedN :: (Quote m) => Int -> m [Dec]
buildFromNestedN n
  | n < 2 = error "n must be at least 2"
  | otherwise = do
      let name = mkName $ "fromNested" ++ show n
      let validateName = mkName $ "validate" ++ show n
      let flattenName = mkName $ "flatten" ++ show (n - 1)
      nested <- newName "nested"
      let params = [VarP nested]
      body <-
        [|
          case $(varE validateName) $(varE nested) of
            Left err -> error err
            Right shape ->
              let array = $(varE flattenName) $(varE nested)
               in Tensor {shape = shape, array = V.fromList array}
          |]
      shapeTyp <- buildTensorShapeType n
      (nestedTyp, a) <- buildNestedNArrayType n
      typ <- [t|$(return nestedTyp) -> Tensor $(return shapeTyp) $(varT a)|]
      let sig = SigD name typ
      let fun = FunD name [Clause params (NormalB body) []]
      return [sig, fun]

buildFromNestedNs :: (Quote m) => Int -> m [Dec]
buildFromNestedNs n = do
  decs <- mapM buildFromNestedN [2 .. n]
  return $ concat decs

buildAll :: (Quote m) => Int -> m [Dec]
buildAll n = do
  validateNs <- buildValidateNs n
  flattenNs <- buildFlattenNs n
  fromNestedNs <- buildFromNestedNs n
  return $ validateNs ++ flattenNs ++ fromNestedNs

-- validate1 :: [a] -> Either String [Int]
-- validate1 l = do
--   when (null l) $ Left "Tensor cannot be empty"
--   return [length l]

-- validate2 :: [[a]] -> Either String [Int]
-- validate2 l = do
--   when (null l) $ Left "Input cannot be empty"
--   nested <- mapM validate1 l
--   when (any (/= head nested) nested) $ Left "All inner lists must have the same length"
--   return $ length l : head nested

-- validate3 :: [[[a]]] -> Either String [Int]
-- validate3 l = do
--   when (null l) $ Left "Input cannot be empty"
--   nested <- mapM validate2 l
--   when (any (/= head nested) nested) $ Left "All inner lists must have the same length"
--   return $ length l : head nested

-- flatten1 :: [[a]] -> [a]
-- flatten1 = concat

-- flatten3 :: [[[a]]] -> [a]
-- flatten3 = concatMap flatten

-- flatten4 :: [[[[a]]]] -> [a]
-- flatten4 = concatMap flatten3

-- fromNested2' :: [[a]] -> Tensor (s1 ': s2) a
-- fromNested2' nested = case validate2 nested of
--   Left err -> error err
--   Right shape ->
--     let array = flatten1 nested
--      in Tensor {shape = shape, array = V.fromList array}

-- fromNested3' :: [[[a]]] -> Tensor [s1, s2, s3] a
-- fromNested3' nested = case validate3 nested of
--   Left err -> error err
--   Right shape ->
--     let array = flatten3 nested
--      in Tensor {shape = shape, array = V.fromList array}
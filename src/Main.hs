import Tensor
import Utils
import qualified Data.Vector as V

main :: IO ()
main = do
    let t1 = Tensor { shape = [2, 3], array = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] } :: Tensor '["dim1", "dim2"] Double
    let t2 = fromList [2, 3] [10.0, 12.0, 13.0, 14.0, 15.0, 16.0] :: DoubleTensor '["dim1", "dim2"]
    let t3 = fromList [1, 3] [100.0, 101.0, 102.0] :: DoubleTensor '["dim1", "dim2"]
    let t4 = fromList [1, 1] [1000.0] :: DoubleTensor '["dim1", "dim2"]
    print t1
    putStrLn $ showBool @(Compatible '["dim"] '["diim"])
    print $ t2 +. t4
    let n = 1
    print $ index t1 (n :~ 1 :~ LNil)
    let t5 = set t1 (0 :~ 1 :~ LNil) 1000.0
    print t5
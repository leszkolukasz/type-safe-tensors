import Tensor
import Utils

main :: IO ()
main = do
    let t1 = Tensor { shape = [2, 3], array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] } :: Tensor '["dim"] Double
    let t2 = fromList [2, 3] [7.0, 8.0, 9.0, 10.0, 11.0, 12.0] :: DoubleTensor '["dim"]
    let t3 = fromList [1, 3] [10.0, 11.0, 12.0] :: DoubleTensor '["dim"]
    print t1
    putStrLn $ showBool @(Compatible '["dim"] '["diim"])
    print $ t1 *. t2
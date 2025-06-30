import Data.Vector qualified as V
import Torch.Tensor
import Torch.Tensor.Types
import Torch.Utils

main :: IO ()
main = do
  let t1 = Tensor {shape = [2, 3], array = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]} :: Tensor '["dim1", "dim2"] Double
  let t2 = fromList [2, 3] [10.0, 12.0, 13.0, 14.0, 15.0, 16.0] :: DoubleTensor '["dim1", "dim2"]
  let t3 = fromList [1, 3] [100.0, 101.0, 102.0] :: DoubleTensor '["dim1", "dim2"]
  let t4 = fromList [1, 1] [1000.0] :: DoubleTensor '["dim1", "dim2"]
  let t42 = fromList [1, 2, 2] [1.0, 0.0, 0.0, 1.0] :: DoubleTensor '["dim1", "dim1", "dim3"]
  let t5 = fromList [3, 2] [100.0, 101.0, 102.0, 103.0, 104.0, 105.0] :: DoubleTensor '["dim2", "dim3"]
  print t1
  putStrLn $ show $ toValue @(Equal (FirstTuple (Split [Any, "Dim1", "Dim2"] 3)) '[Any, "Dim1", "Dim2"])
  putStrLn $ show $ toValue @(Equal (Unsqueeze '["Dim1", "Dim2"] 1) ["Dim1", Any, "Dim2"])
  putStrLn $ show $ toValue @(Broadcastable '["Dim1", "Dim2"] '["Dim3", "Dim1", "Dim3"])
  print $ t2 +. t4
  let n = 1
  print $ get t1 (n :~ 1 :~ LNil)
  let t6 = set t1 (0 :~ 1 :~ LNil) 1000.0
  print $ t6
  -- let t7 = t4 @. t5
  -- print t7
  -- print (t7 +. t7)
  print $ squeeze @0 t3
  let t8 = fromList [3] [1000.0, 10001.0, 10002.0] :: DoubleTensor '["dim3"]
  let t9 = fromList [1, 1, 3] [1000.0, 10001.0, 10002.0] :: DoubleTensor '["dim1", "dim2", "dim3"]
  print $ t9 +. unsqueezeTo t8 t9
  print $ squeeze @0 $ slice t1 (Single 1 :~ Range 1 2 :~ LNil)
  let t10 = fromList [2, 2] [1.0, 0.0, 1.0, 1.0] :: DoubleTensor '["dim1", "dim1"]
  -- print $ t10 @. t42
  print $ t1 *. t4
  print $ (fromNested2 [[1.0, 2.0], [3.0, 4.0]] :: DoubleTensor '["dim1", "dim2"])
  print $ (fromList [2, 2, 2] [13.0, 13.0, 14.0, 14.0, 15.0, 15.0, 16.0, 16.0] :: DoubleTensor '["dim1", "dim2", "dim3"])
  print $ t1
  print $ reshapeUnsafe t1 [3, 2]
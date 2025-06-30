import Data.List (intercalate)
import Data.Vector qualified as V
import System.IO (hFlush, stdout)
import Torch.Module
import Torch.Tensor
import Torch.Utils

type SIZE = "SIZE"

type BATCH = "BATCH"

type FEAT = "FEAT"

type FC1_OUT = "FC1_OUT"

type FC2_IN = "FC2_IN"

type FC2_OUT = "FC2_OUT"

main :: IO ()
main = do
  fc1_weight :: DoubleTensor '[FEAT, FC1_OUT] <- fromFile "./pytorch/fc1_weight.txt"
  fc1_bias :: DoubleTensor '[FC1_OUT] <- fromFile "./pytorch/fc1_bias.txt"

  fc2_weight :: DoubleTensor '[FC1_OUT, FC2_OUT] <- fromFile "./pytorch/fc2_weight.txt"
  fc2_bias :: DoubleTensor '[FC2_OUT] <- fromFile "./pytorch/fc2_bias.txt"

  let fc1 = Linear {weight = fc1_weight, bias = fc1_bias}
  let fc2 = Linear {weight = fc2_weight, bias = fc2_bias}

  putStr "Image index:"
  hFlush stdout
  imgIdx :: Int <- read <$> getLine
  input :: DoubleTensor '[SIZE, SIZE] <- fromFile ("./pytorch/mnist_exports/image_" ++ show imgIdx ++ ".txt")

  let flatInput = reshapeUnsafe @'[BATCH, FEAT] input [-1]
  let out1 = linear fc1 flatInput
  let act1 = relu out1
  let logits = linear fc2 act1
  let probs = softmax logits |> array |> V.toList |> map (\x -> round (x * 100) :: Int)
  putStr $ intercalate ", " (zip [0 .. 9] probs |> map (\(i, p) -> show i ++ ": " ++ show p ++ "%"))
import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

def export_weights():
    state_dict = torch.load("model.pth", weights_only=True)

    for key in state_dict.keys():
        param = state_dict[key].t()
        shape = param.shape
        values = param.flatten()

        name = key.replace('.', '_')

        with open(f"{name}.txt", "w") as f:
            f.write(" ".join(str(dim) for dim in shape) + "\n")
            f.write(" ".join(str(v.item()) for v in values) + "\n")

def export_mnist_image(idx, folder="mnist_exports"):
    os.makedirs(folder, exist_ok=True)
    
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    image, label = mnist[idx]
    
    tensor = image.squeeze(0)
    shape = tensor.shape
    values = tensor.flatten()
    
    filename = os.path.join(folder, f"image_{idx}.txt")
    with open(filename, "w") as f:
        f.write(" ".join(str(dim) for dim in shape) + "\n")
        f.write(" ".join(str(v.item()) for v in values) + "\n")

    filename_png = os.path.join(folder, f"image_{idx}_label_{label}.png")
    img = transforms.ToPILImage()(image)
    img.save(filename_png)

if __name__ == "__main__":
    export_weights()
    
    for i in range(100):
        export_mnist_image(i)
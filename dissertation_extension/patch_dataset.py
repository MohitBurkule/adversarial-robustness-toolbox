import os
import urllib.request
import tarfile
import torch
import torchvision
from torchvision import datasets, transforms

# Configurable dataset name from environment variable
DATASET_NAME = os.environ.get("PATCH_DATASET_NAME", "cifar10").lower()
DATA_ROOT = "./data"

print(f"[*] Patching torchvision.datasets.FashionMNIST to redirect to: {DATASET_NAME.upper()}")

def download_imagenette(root_dir=DATA_ROOT):
    os.makedirs(root_dir, exist_ok=True)
    tar_path = os.path.join(root_dir, "imagenette2-160.tgz")
    extracted_dir = os.path.join(root_dir, "imagenette2-160")
    
    if not os.path.exists(extracted_dir):
        if not os.path.exists(tar_path):
            print(f"[*] Downloading Imagenette-160 from Fast.ai...")
            url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
            urllib.request.urlretrieve(url, tar_path)
            print("[*] Download complete.")
            
        print("[*] Extracting Imagenette-160...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=root_dir)
        print("[*] Extraction complete.")
        
    return extracted_dir

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
        
        # Preprocessing to convert RGB/color images to 28x28 grayscale images
        # so they perfectly fit the Fashion-MNIST architectures in the hypotheses.
        self.pre_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28))
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        # Apply domain mapping to 1-channel, 28x28 grayscale
        img = self.pre_transform(img)
        
        # Apply the original script's transform if present
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label

def get_patched_dataset(root, train=True, download=True, transform=None):
    if DATASET_NAME == "cifar10":
        # Load CIFAR-10 natively
        base_dataset = datasets.CIFAR10(root=root, train=train, download=download, transform=None)
        return DatasetWrapper(base_dataset, transform=transform)
        
    elif DATASET_NAME == "imagenette":
        # Download and load Imagenette-160 using ImageFolder
        extracted_dir = download_imagenette(root_dir=root)
        split_dir = "train" if train else "val"
        base_dataset = datasets.ImageFolder(root=os.path.join(extracted_dir, split_dir), transform=None)
        return DatasetWrapper(base_dataset, transform=transform)
        
    else:
        # Fallback to Fashion-MNIST if not matched
        print(f"[!] Warning: Unknown patched dataset name '{DATASET_NAME}'. Falling back to original FashionMNIST.")
        return datasets.OriginalFashionMNIST(root=root, train=train, download=download, transform=transform)

# Store the original class in case a script explicitly needs it or we need a fallback
if not hasattr(datasets, "OriginalFashionMNIST"):
    datasets.OriginalFashionMNIST = datasets.FashionMNIST

# Monkey-patch FashionMNIST!
datasets.FashionMNIST = get_patched_dataset
torchvision.datasets.FashionMNIST = get_patched_dataset

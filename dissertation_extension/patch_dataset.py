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


# ---------------------------------------------------------------------------
# Modular arithmetic dataset
#
# All 157 hypothesis scripts hardcode N_CLASSES = 10 and CNN(n=10), so this
# dataset must have exactly 10 classes.  We use (a + b) mod 10 where
# a, b ∈ {0 … 99}, giving 10 000 unique (a, b) pairs, perfectly balanced
# (1 000 samples per class).  Train / test split: 8 000 / 2 000.
#
# Each sample is encoded as a 1×28×28 float32 tensor:
#   pixels   0 –  99  : one-hot for a   (a-th pixel = 1.0)
#   pixels 100 – 199  : one-hot for b   (100+b-th pixel = 1.0)
#   pixels 200 – 783  : 0.0
# Reshape to (1, 28, 28).
#
# NOTE ON GROKKING: The classical grokking result (Power et al., 2022) uses
# p=97 (97 classes) and trains on only 30–40 % of samples for thousands of
# epochs with high weight-decay.  That setup is incompatible with the
# N_CLASSES=10 / 10-epoch scripts here.  For grokking-specific analysis see
# H157 (training trajectory) and H160 (CIFAR-100 long run).  This dataset
# still lets you study adversarial vulnerability on a *structured mathematical
# task*, which is interesting in its own right.
# ---------------------------------------------------------------------------

class ModularArithmeticDataset(torch.utils.data.Dataset):
    """
    (a + b) mod 10 over a=0..99, b=0..99.
    Encoded as 1×28×28 one-hot float tensors; 10 classes.
    Train split: first 8 000 pairs (sorted); test split: last 2 000 pairs.
    """
    IMG_SIZE = 28 * 28   # 784

    def __init__(self, train: bool = True):
        super().__init__()
        # All 10 000 unique (a, b) pairs in a fixed deterministic order
        pairs = [(a, b) for a in range(100) for b in range(100)]
        # Shuffle with fixed seed so every run is identical
        rng = torch.Generator()
        rng.manual_seed(2022)
        idx = torch.randperm(10_000, generator=rng).tolist()
        pairs = [pairs[i] for i in idx]

        if train:
            self.pairs = pairs[:8_000]
        else:
            self.pairs = pairs[8_000:]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        label = (a + b) % 10

        img = torch.zeros(self.IMG_SIZE)
        img[a] = 1.0          # one-hot for a  in pixels   0–99
        img[100 + b] = 1.0    # one-hot for b  in pixels 100–199
        img = img.view(1, 28, 28)   # (C=1, H=28, W=28) – matches CNN input

        return img, label

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
    def __init__(self, base_dataset, transform=None, is_train=True):
        self.base_dataset = base_dataset
        self.transform = transform
        self.is_train = is_train
        
        # Preprocessing to convert RGB/color images to 28x28 grayscale images
        # so they perfectly fit the Fashion-MNIST architectures in the hypotheses.
        self.pre_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28))
        ])
        
        # To prevent CUDA Out-Of-Memory (OOM) on large test sets (e.g. SVHN test set is 26,032 images)
        # when scripts attempt to stack the entire test set into a single tensor on the GPU VRAM,
        # we truncate the test set to a maximum of 10,000 samples.
        self.max_test_samples = 10000
        if not self.is_train and len(self.base_dataset) > self.max_test_samples:
            print(f"[*] Truncating large test dataset from {len(self.base_dataset)} to {self.max_test_samples} to prevent GPU CUDA OOM.")
            self.dataset_len = self.max_test_samples
        else:
            self.dataset_len = len(self.base_dataset)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if idx >= self.dataset_len:
            raise IndexError("Index out of bounds for truncated dataset.")
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
        return DatasetWrapper(base_dataset, transform=transform, is_train=train)
        
    elif DATASET_NAME == "imagenette":
        # Download and load Imagenette-160 using ImageFolder
        extracted_dir = download_imagenette(root_dir=root)
        split_dir = "train" if train else "val"
        base_dataset = datasets.ImageFolder(root=os.path.join(extracted_dir, split_dir), transform=None)
        return DatasetWrapper(base_dataset, transform=transform, is_train=train)
        
    elif DATASET_NAME == "svhn":
        # Street View House Numbers (10 classes of real-world digits)
        split_name = "train" if train else "test"
        base_dataset = datasets.SVHN(root=root, split=split_name, download=download, transform=None)
        return DatasetWrapper(base_dataset, transform=transform, is_train=train)
        
    elif DATASET_NAME == "stl10":
        # STL-10 (ImageNet-like animal/vehicle images, 10 classes, 96x96)
        split_name = "train" if train else "test"
        base_dataset = datasets.STL10(root=root, split=split_name, download=download, transform=None)
        return DatasetWrapper(base_dataset, transform=transform, is_train=train)
        
    elif DATASET_NAME == "kmnist":
        # Kuzushiji-MNIST (10 classes of ancient Japanese hiragana)
        base_dataset = datasets.KMNIST(root=root, train=train, download=download, transform=None)
        return DatasetWrapper(base_dataset, transform=transform, is_train=train)
        
    elif DATASET_NAME in ("modular_arithmetic", "mod_arith", "modular"):
        # (a + b) mod 10 over a,b ∈ {0…99}; 10 classes; no download needed
        return ModularArithmeticDataset(train=train)

    elif DATASET_NAME == "eurosat":
        # EuroSAT (10 classes of satellite land cover classification)
        base_dataset = datasets.EuroSAT(root=root, download=download, transform=None)
        generator = torch.Generator().manual_seed(42)
        train_len = int(0.8 * len(base_dataset))
        test_len = len(base_dataset) - train_len
        train_sub, test_sub = torch.utils.data.random_split(base_dataset, [train_len, test_len], generator=generator)
        selected_subset = train_sub if train else test_sub
        return DatasetWrapper(selected_subset, transform=transform, is_train=train)
        
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

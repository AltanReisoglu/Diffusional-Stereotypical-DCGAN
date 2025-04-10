from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import kagglehub
from PIL import Image
from torch.utils.data import DataLoader, random_split
import os
# Download latest version
path = kagglehub.dataset_download("arnaud58/landscape-pictures")

IMG_SIZE = 128
transform = transforms.Compose([
    
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    # Convert images to PyTorch tensors and scale to [0, 1]
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class FlatImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = [os.path.join(folder_path, fname)
                            for fname in os.listdir(folder_path)
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img
    


dataset=FlatImageDataset(folder_path=path,transform=transform)

train_dataloaded=DataLoader(dataset,32,shuffle=True,drop_last=True)


import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import random
from torchvision import datasets


class Hindi_Digits(Dataset):
    def __init__(self, csv='LabelMap.csv'):
        self.annotation = pd.read_csv(csv)
        self.transform = transforms.Compose([
                          transforms.ToPILImage(),
                          transforms.Resize(256),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5), (0.5))])
        self.label_dict = self.generate_label_dict()


    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.annotation.iloc[idx].path
        label = self.annotation.iloc[idx].label
        image = Image.open(img_name.item())
        image = transforms.PILToTensor()(image)
        
        siblings = self.get_siblings(label)
        triplet = torch.vstack([image.unsqueeze(0), siblings])
        
        img_triplet = torch.zeros([3, 256, 256])
        for i,img in enumerate(triplet):
            img_triplet[i] = self.transform(img.to(torch.uint8))
        return img_triplet, label

    def generate_label_dict(self):
        df  = self.annotation
        hindi_dict = {}
        for i in range(0,10):
            image_list = df.index[df['label']==i].tolist()
            hindi_dict[i]=image_list
        return hindi_dict

    def get_siblings(self, label):
        image_list = self.label_dict[label.item()]
        siblings = []
        for i in range(2):
            rand_index = random.sample(image_list,1)
            image = Image.open(self.annotation.iloc[rand_index]['path'].item())
            image = transforms.PILToTensor()(image)
            image = image.to(torch.float32)
            siblings.append(image.unsqueeze(0))
        return torch.vstack(siblings)

class Custom_MNIST(Dataset):
    def __init__(self):
        
        self.transform = transforms.Compose([
                          transforms.ToPILImage(),
                          transforms.Resize(256),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5), (0.5))])
        self.dataset = datasets.MNIST(root="", download=True, transform=None)
        self.label_dict = self.generate_label_dict()


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        image = self.dataset.data[idx]
        label = self.dataset.targets[idx]


        siblings = self.get_siblings(label)
        triplet = torch.vstack([image.unsqueeze(0), siblings])
        
        img_triplet = torch.zeros([3, 256, 256])
        for i,img in enumerate(triplet):
            img_triplet[i] = self.transform(img)

        return img_triplet, label

    def generate_label_dict(self):
        mnist_dict = {}
        for i in range(0,10):
            indices = self.dataset.targets == i # if you want to keep images with the label 5
            mnist_dict[i]=self.dataset.data[indices]
        return mnist_dict

    def get_siblings(self, label):
        img_loader = torch.utils.data.DataLoader(dataset=self.label_dict[label.item()],
                                               batch_size=2,
                                               shuffle=True,
                                               )
        
        for siblings in img_loader:
            return siblings
         
class combo_dataset(Dataset):
    def __init__(self, csv='LabelMap.csv'):
        self.mnist = Custom_MNIST()
        self.hindi = Hindi_Digits()

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        return self.mnist[idx], self.hindi[idx]

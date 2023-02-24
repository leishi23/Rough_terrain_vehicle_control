import json
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os

transform = transforms.Compose([
    transforms.PILToTensor(),
    # transforms.ToTensor(),
])

class carla_rgb(Dataset):
    
    def __init__(self, file_path):

        super().__init__()
        self.x = datasets.ImageFolder(file_path, transform=transform)

    def __getitem__(self, index):

        assert index < len(self.x)
        return self.x[index]

    def __len__(self):

        return len(self.x)
    
    
class carla_vec(Dataset):
    
    def __init__(self, file_path):
        
        super().__init__()
        with open(file_path) as f:
            data = json.load(f)
        self.x = data['Vector_output']
    
    def __getitem__(self, index):

        assert index < len(self.x)
        return self.x

    def __len__(self):

        return len(self.x)
        

class carla_action(Dataset):
    
    def __init__(self, file_path, horizon=8):
        
        super().__init__()
        with open(file_path) as f:
            raw_data = json.load(f)['Action_input']
            datapoint = []
            data = []
            for i in range(len(raw_data)):
                if i % horizon == 0 and i != 0: 
                    data.append(datapoint)
                    datapoint = []
                datapoint.append(raw_data[i])
                if i == len(raw_data)-1:        # to add the last datapoint
                    data.append(datapoint)
                                                  
        self.x = data
    
    def __getitem__(self, index):

        assert index < len(self.x)
        return self.x

    def __len__(self):

        return len(self.x)

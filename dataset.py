from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor,PILToTensor, Compose,Resize
from torch.utils.data import DataLoader
import os


class MyDataset(Dataset):
    def __init__(self,root, train= True,transform=None):
        self.root = root
        self.transform = transform 
       
        # list dir
        path = os.listdir(self.root)
        dir = [os.path.join(root,i) for i in path]
            
        #list image files on dir
        bike = [os.path.join(dir[0],file) for file in os.listdir(dir[0])] #
        car = [os.path.join(dir[1],file) for file in os.listdir(dir[1])] #
        
        #creat len to split data
        len_bike = len(bike)
        len_car = len(car)

        #creat warehouse to store data
        self.images = []
        self.labels = []

        if train:  #when train
            
            for i in range(int(len_bike*0.8)):
                image = self.transform(Image.open(bike[i]).convert("RGB"))
                self.images.append(image)
                self.labels.append('bike')
            for i in range(int(len_car*0.8)):
                image = self.transform(Image.open(car[i]).convert("RGB"))
                self.images.append(image)
                self.labels.append('car')
        else:   #when test
            for i in range(int(len_bike*0.8),len_bike):
                image = self.transform(Image.open(bike[i]).convert("RGB"))
                self.images.append(image)
                self.labels.append('bike')
            for i in range(int(len_car*0.8),len_car):
                image = self.transform(Image.open(car[i]).convert("RGB"))
                self.images.append(image)
                self.labels.append('car') 
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]

if __name__ == '__main__':
    
    transform = Compose([
                Resize((224, 224)), 
                ToTensor()
        ])
    test_data = MyDataset(root="D:\car and moto\Car-Bike-Dataset", train=False,transform=transform)

    

    test_data = DataLoader(
       dataset= test_data,
       batch_size= 8,
       num_workers= 4,
       shuffle= False,
       drop_last= True              
    )
    for i,j in test_data:
        print(i.shape) 


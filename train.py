from model import MyModel
from dataset import MyDataset
from torch.utils.data import DataLoader 
from torchvision.transforms import Compose,Resize, ToTensor

if __name__ == '__main__':
    transform = Compose([
                        Resize((224, 224)), 
                        ToTensor()
                ])
    train_data = MyDataset(root="D:\car and moto\Car-Bike-Dataset", train=True,transform=transform)
    test_data = MyDataset(root="D:\car and moto\Car-Bike-Dataset", train=False,transform=transform)
    train_data = DataLoader(
            dataset= train_data,
            batch_size=8,
            num_workers=4,
            shuffle=True,
            drop_last= False
        ) 

            
    test_data = DataLoader(
            dataset= test_data,
            batch_size= 8,
            num_workers= 4,
            shuffle= False,
            drop_last= True              
            )

    for i,j in test_data:
        print(i.shape)



from dataset import MyDataset
from model import MyModel
from torch.utils.data import DataLoader
from torchvision.transforms import Compose,Resize, ToTensor
import torch.nn as nn
import torch

if __name__ == '__main__':

    transform = Compose([
                        Resize((224,224)),
                        ToTensor()
                ])

    num_epochs= 10

    train_data = MyDataset(root="/root/.cache/kagglehub/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset/versions/1/Car-Bike-Dataset", train=True,transform=transform)
    test_data = MyDataset(root="/root/.cache/kagglehub/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset/versions/1/Car-Bike-Dataset", train=False,transform=transform)

    train_data = DataLoader(
            dataset= train_data,
            batch_size=8,
            num_workers=4,
            shuffle=True,
            drop_last= True
        )


    test_data = DataLoader(
            dataset= test_data,
            batch_size= 8,
            num_workers= 4,
            shuffle= False,
            drop_last= True
            )

    model = MyModel()
    criterion =nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum= 0.9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for image, label in train_data:
            if torch.cuda.is_available():
                image=image.to(device)
                label=label.to(device)

            output = model(image)
            loss_values = criterion(output,label)

            optimizer.zero_grad()
            loss_values.backward()
            optimizer.step()

        model.eval()
        predict =[]
        labels = []
        for image, label in test_data:

            image=image.to(device)
            label=label.to(device)

            with torch.no_grad():
                output = model(image)
                indice = torch.argmax(output.cpu(), dim=1)
                predict.append(indice)
                labels.append(label)
                loss_values = criterion(output,label)
        all_predict = []
        all_label = []
        for i in predict:
          all_predict.extend(i.tolist())
        for j in labels:
          all_label.extend(j.tolist())
        print(all_predict)
        print(all_label)
        exit(0)

from dataset import MyDataset
from model import MyModel 
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from torchvision.transforms import Compose,Resize, ToTensor
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pickle

if __name__ == '__main__':

    #creat transform
    transform = Compose([
                        Resize((224,224)),
                        ToTensor()
                ])
    num_epochs= 10
    
    #creat train, test data
    train_data = MyDataset(root="/kaggle/input/car-vs-bike-classification-dataset/Car-Bike-Dataset", train=True,transform=transform)
    test_data = MyDataset(root="/kaggle/input/car-vs-bike-classification-dataset/Car-Bike-Dataset", train=False,transform=transform)
    
    #train data loader to fix model
    train_data = DataLoader(
            dataset= train_data,
            batch_size=8,
            num_workers=4,
            shuffle=True,
            drop_last= True
        )

    #test data loader to test model
    test_data = DataLoader(
            dataset= test_data,
            batch_size= 8,
            num_workers= 4,
            shuffle= False,
            drop_last= True
            )
    #sumary to store infomation about model
    writer = SummaryWriter('/content/sample_data')
    #creat model
    model = MyModel()
    #import loss function
    criterion =nn.CrossEntropyLoss()
    #import optimize function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum= 0.9)
    #check device if it have GPU return cuda, if not, return cpu 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #push model on cuda if it have
    model = model.to(device)
    num_inter = len(train_data)
    #training
    for epoch in range(num_epochs):
        model.train()
        #creat task bar on terminal
        bar = tqdm(train_data)
        #forward
        for inter,(image, label) in enumerate(bar):
            #push image and label GPU 
            if torch.cuda.is_available():
                image=image.to(device)
                label=label.to(device)

            output = model(image)
            loss_values = criterion(output,label)
            #record loss fuction on sumary
            writer.add_scalar("train/loss", loss_values.item(),epoch+num_inter + inter)
            #backward
            optimizer.zero_grad()
            loss_values.backward()
            optimizer.step()
        #not train
        model.eval()
        #creat list to store prediction and label
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
        #add prediction and label from tensor to number of list
        for i in predict:
          all_predict.extend(i.tolist())
        for j in labels:
          all_label.extend(j.tolist())
        #print accuracy, recall,.. after epoch
        print(classification_report(all_label,all_predict))
        accuracy = accuracy_score(all_label,all_predict)
        #record accuracy after test on sumary
        writer.add_scalar("test/accuracy", accuracy,epoch)
    
    #save model
    torch.save(model.state_dict(), "model.pth")
    
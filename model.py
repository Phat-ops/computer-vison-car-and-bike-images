import torch.nn as nn
import torch

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = self.make(in_channels=3, out_channels=8)
        self.cnn2 = self.make(in_channels=8, out_channels=16)
        self.cnn3 = self.make(in_channels=16, out_channels=32)
        self.cnn4 = self.make(in_channels=32, out_channels=64)
        self.cnn5 = self.make(in_channels=64, out_channels=128)

        self.fatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Linear(in_features= 6272, out_features= 512),
            nn.LeakyReLU() 
        )
        self.fc2 = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Linear(in_features= 512, out_features= 256),
            nn.LeakyReLU() 
        )
        self.fc3 = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Linear(in_features= 256, out_features= 2)
        )

    def make(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size= 3,stride=1,padding='same'),
            nn.BatchNorm2d(num_features= out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size= 3,stride=1,padding='same'),
            nn.BatchNorm2d(num_features= out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
    
    def forward(self,x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.fatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
if __name__ == '__main__':
    pass

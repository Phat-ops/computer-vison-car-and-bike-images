import torch.nn as nn
import torch


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
                nn.Linear(in_features=3*224*224, out_features= 64),
                nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
                nn.Linear(in_features=64, out_features= 2),
        )


    def forward(self,x):
        x = self.flatten(x)
        x = self.fc1(x)
        x= self.fc2(x)
        return x
if __name__ == '__main__':
    pass

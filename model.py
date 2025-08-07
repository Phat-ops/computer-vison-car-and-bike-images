import torch.nn as nn
import torch


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=3*5, out_features= 32)
        self.ac = nn.LeakyReLU()
    def forward(self,x):
        x = self.flatten(x)
        x = self.fc(x)
        x= self.ac(x)
        return x
if __name__ == '__main__':
    model = MyModel()
    input = torch.rand(8,3,5)
    result = model(input)
    print(result.shape)
    
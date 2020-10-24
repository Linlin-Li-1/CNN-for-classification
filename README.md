# CNN Model for Classification

I used PyTorch to train a CNN model to classify images of three insects (beetles, cockroaches and dragonflies).


The structure of the model is as follow:
- input size: (3 x 84, 84); 
- conv1: Conv2d(3, 16, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)); 
- max pool 2d
- conv2: Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1)); 
- fc1: Linear(in_features=5184, out_features=256, bias=True);
- fc2: Linear(in_features=256, out_features=84, bias=True);
- fc3: Linear(in_features=84, out_features=3, bias=True).
)

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)
        self.fc1   = nn.Linear(in_features=64 * 9 * 9, out_features=256) 
        self.fc2   = nn.Linear(in_features=256, out_features=84)
        self.fc3   = nn.Linear(in_features=84, out_features=out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

![cnn](cnn.png)

Data Source: https://www.insectimages.org/index.cfm. 

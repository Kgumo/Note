```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```


```python
train_dataset=datasets.MNIST(
    root='F:/0.Project/0.VS\Py_project/DP/MNIST_CNN/MNIST_data/MNIST/raw',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
train_loader=DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
test_dataset=datasets.MNIST(
    root='F:/0.Project/0.VS\Py_project/DP/MNIST_CNN/MNIST_data/MNIST/raw',
    train=False,
    download=False,
    transform=transforms.ToTensor()
)
test_loader=DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)
```


```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(28*28,128)
        self.layer2=nn.Linear(128,64)
        self.layer3=nn.Linear(64,10)
        
        self.dropout1=nn.Dropout(0.2)
        self.dropout2=nn.Dropout(0.2)
    def forward(self,x):
        x=x.view(-1,28*28)
        x=torch.relu(self.layer1(x))
        x=self.dropout1(x)
        x=torch.relu(self.layer2(x))
        x=self.dropout2(x)
        x=self.layer3(x)
        return x
```


```python
model=MyModel()

fake_images=torch.randn(64,1,28,28)
output=model(fake_images)
print(f"输入形状: {fake_images.shape}")
print(f"输出形状: {output.shape}")
```

    输入形状: torch.Size([64, 1, 28, 28])
    输出形状: torch.Size([64, 10])
    


```python
import torch.optim as optim
num_epochs=5
criterion=nn.CrossEntropyLoss()
learning_rate=0.001
optimizer=optim.Adam(model.parameters(),lr=learning_rate)
device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    
    running_loss=0.0
    correct=0
    total=0
    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs=model(images)
        loss=criterion(outputs,labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss+=loss.item()
        avg_loss=running_loss/len(train_loader)
        
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
        acrruacy=100*correct/total
        print(f"Epoch [{epoch+1}/{num_epochs}],Loss:{avg_loss:.4f},Accuracy:{acrruacy:.2f}%")
```


```python
model.eval()
with torch.no_grad():
    correct=0
    total=0
    test_loss=0.0
    for images,labels in test_loader:
        images =images.to(device)
        labels =labels.to(device)
        outputs=model(images)
        loss=criterion(outputs,labels)
        test_loss+=loss.item()
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    accuracy=100*correct/total
    avg_loss=test_loss/len(test_loader)
    print(f"测试集上的Loss:{avg_loss:.4f},Accuracy:{accuracy:.4f}%")
        
```

    测试集上的Loss:0.0822,Accuracy:97.6600%
    


```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,32,kernel_size=3)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(32,64,kernel_size=3)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(64*5*5,128)
        self.fc2=nn.Linear(128,10)
        self.dropout=nn.Dropout(0.25)
        
    def forward(self,x):
        x=self.pool1(torch.relu(self.conv1(x)))
        x=self.pool2(torch.relu(self.conv2(x)))
        x=x.view(-1,64*5*5)
        x=torch.relu(self.fc1(x))
        x=self.dropout(x)
        x=self.fc2(x)
        return x
model=CNN()
```


```python
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(
    model.parameters(),
    lr=0.001)
model.to(device)
num_epochs=5
model.train()
for epoch in range(num_epochs):
    running_loss=0.0
    correct=0
    total=0
    for images,labels in train_loader:
        optimizer.zero_grad()
        
        labels=labels.to(device)
        images=images.to(device)
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        running_loss+=loss.item()
        avg_loss=running_loss/len(train_loader)
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    accuracy=100*correct/total
    print(f"Epoch[{epoch+1}/{num_epochs}],loss:{avg_loss:.4f},Accuracy:{accuracy:.2f}%")
```

    Epoch[1/5],loss:0.2266,Accuracy:0.05%
    Epoch[2/5],loss:0.0722,Accuracy:0.05%
    Epoch[3/5],loss:0.0546,Accuracy:0.05%
    Epoch[4/5],loss:0.0422,Accuracy:0.05%
    Epoch[5/5],loss:0.0383,Accuracy:0.05%
    


```python
with torch.no_grad():
    correct=0
    total=0
    test_loss=0.0
    for images,labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        loss=criterion(outputs,labels)
        test_loss+=loss.item()
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    avg_loss=test_loss/len(test_loader)
    accuracy=100*correct/total
    print(f"test[loss:{avg_loss:.4f},Accuracy:{accuracy:.2f}]")
```

    test[loss:0.0450,Accuracy:98.67]
    

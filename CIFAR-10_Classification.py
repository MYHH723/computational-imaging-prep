import torch as t
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torch import optim
show = ToPILImage()

if __name__ == '__main__':

    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = tv.datasets.CIFAR10(
        root='./pytorch-book-cifar10/',
        train=True,
        download=True,
        transform=transform)

    trainloader = t.utils.data.DataLoader(
        trainset,
        batch_size=4,
        shuffle=True,
        num_workers=0)

    testset = tv.datasets.CIFAR10(
        './pytorch-book-cifar10/',
        train=False,
        download=True,
        transform=transform)

    testloader = t.utils.data.DataLoader(
        testset,
        batch_size=4,
        shuffle=False,
        num_workers=0)

        
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck')

    print("训练集图片数量：", len(trainset))
    print("测试集图片数量：", len(testset))

    (data, label) = trainset[100]
    print(classes[label])

    #show((data + 1) / 2).resize((100, 100)).show()

    dataiter = iter(trainloader)
    images, labels = next(dataiter) #Older version PyTorch uses 'dataiter.next()'
    print(' '.join('%11s'%classes[labels[j]] for j in range(4)))
    #show(tv.utils.make_grid((images + 1) / 2)).resize((400,100)).show()

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)  
            self.fc1   = nn.Linear(16 * 5 * 5, 120)  
            self.fc2   = nn.Linear(120, 84)
            self.fc3   = nn.Linear(84, 10)

        def forward(self, x): 
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) 
            x = F.max_pool2d(F.relu(self.conv2(x)), 2) 
            x = x.view(x.size()[0], -1) 
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)        
            return x

    net = Net()
    net = net.to(device)
    print(net)

    #Define Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #Train the network
    '''
    The three main steps in tranning NN:

        1. input data

        2. Forward and Backward propagation

        3. Update parameters
    '''
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' \
                        % (epoch+1, i+1, running_loss / 2000))
                    running_loss = 0.0
    print("Finished Training")

    dataiter = iter(testloader)
    images, labels = next(dataiter) 

    show(tv.utils.make_grid(images / 2 - 0.5)).resize((400, 100)).show()

    images = images.to(device)
    outputs = net(images)

    _, predicted = t.max(outputs.data, 1)

    print('实际的label: ', ' '.join('%08s'%classes[labels[j]] for j in range(4)))
    print('预测结果: ', ' '.join('%5s'% classes[predicted[j].item()] for j in range(4)))

    correct = 0
    total = 0

    with t.no_grad():
         for data in testloader:
              images, labels = data
              images = images.to(device)
              labels = labels.to(device)
              outputs = net(images)
              _, predicted = t.max(outputs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
    print('10000张测试集中的准确率为: %f %%' % (100 * correct // total))
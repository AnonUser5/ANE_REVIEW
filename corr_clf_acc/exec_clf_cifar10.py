# Test accuracy > 77%


import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from util.models import CNN


n_epochs = 300
batch_size = 128


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
dataset_train = datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform_train)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataset_test = datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


model = CNN(n_classes=10, n_channels=3).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
criterion = torch.nn.CrossEntropyLoss()

for i in range(0, n_epochs):
    train_loss = model.fit(loader_train, optimizer, criterion)
    print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(i + 1, n_epochs, train_loss))

print('Test accuracy: {:.2f}'.format(model.eval_acc(loader_test)))



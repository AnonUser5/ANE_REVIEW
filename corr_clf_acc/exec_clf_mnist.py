# Test accuracy > 99%


import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from util.models import CNN


n_epochs = 100
batch_size = 128


dataset_train = datasets.MNIST(root='../../data/', train=True, transform=transforms.ToTensor(), download=True)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataset_test = datasets.MNIST(root='../../data/', train=False, transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

model = CNN(n_classes=10).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
criterion = torch.nn.CrossEntropyLoss()

for i in range(0, n_epochs):
    train_loss = model.fit(loader_train, optimizer, criterion)
    print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(i + 1, n_epochs, train_loss))

print('Test accuracy: {:.2f}'.format(model.eval_acc(loader_test)))



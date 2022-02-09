import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class CNN(nn.Module):
    def __init__(self, n_classes=10, n_channels=1):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        if n_channels == 1:
            self.fc1 = nn.Sequential(
                nn.Linear(7 * 7 * 32, 100, bias=False),
                nn.ReLU())
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(8 * 8 * 32, 100, bias=False),
                nn.ReLU())
        self.fc2 = nn.Linear(100, n_classes, bias=False)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        train_loss = 0

        for i, (images, labels) in enumerate(data_loader):
            images = images.cuda()
            labels = labels.cuda()

            preds = self(images)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        return train_loss / len(data_loader)

    def eval_acc(self, data_loader):
        self.eval()

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in data_loader:
                images = images.cuda()
                labels = labels.cuda()

                preds = self(images)
                _, predicted = torch.max(preds, 1)
                correct += (predicted == labels).sum().item()
                total += labels.shape[0]

        return 100 * (correct / float(total))


class GCN(nn.Module):
    def __init__(self, n_node_feats, n_classes):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(n_node_feats, 128)
        self.gc2 = GCNConv(128, 128)
        self.gc3 = GCNConv(128, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, g):
        h = F.relu(self.gc1(g.x, g.edge_index))
        h = F.relu(self.gc2(h, g.edge_index))
        h = F.relu(self.gc3(h, g.edge_index))
        hg = global_mean_pool(h, g.batch)
        h = F.relu(self.fc1(hg))
        out = self.fc2(h)

        return out

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        train_loss = 0

        for batch in data_loader:
            batch.batch = batch.batch.cuda()

            preds = self(batch)
            loss = criterion(preds, batch.y.flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def eval_acc(self, data_loader):
        self.eval()

        with torch.no_grad():
            correct = 0
            total = 0

            for batch in data_loader:
                batch.batch = batch.batch.cuda()

                preds = self(batch)
                _, predicted = torch.max(preds, 1)
                correct += (predicted == batch.y.flatten()).sum().item()
                total += batch.y.shape[0]

        return 100 * (correct / float(total))

import torch
from torch_geometric.data import DataLoader
from util.crystal import load_dataset
from util.crystal import split_dataset
from util.models import GCN


n_epochs = 300
batch_size = 128


dataset = load_dataset(path='../../chem_data/crystal/mpl',
                       metadata_file='../../chem_data/crystal/mpl/metadata_bg_clf.xlsx',
                       idx_target=2)
dataset_train, dataset_test = split_dataset(dataset, ratio=0.8)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(dataset_test, batch_size=batch_size)


model = GCN(n_node_feats=dataset[0].x.shape[1], n_classes=10).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
criterion = torch.nn.CrossEntropyLoss()

for i in range(0, n_epochs):
    train_loss = model.fit(loader_train, optimizer, criterion)
    print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(i + 1, n_epochs, train_loss))

print('Test accuracy: {:.2f}'.format(model.eval_acc(loader_test)))

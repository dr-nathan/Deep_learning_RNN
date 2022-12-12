import itertools

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from data_rnn import load_imdb
from tqdm import tqdm

device = torch.device('cpu')

# load iMDB data
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=True) # for val set final to false
# every x is a list of words, every y is a single int label

# add start and end tokens to every x
x_train = [[w2i['.start']] + x + [w2i['.end']] for x in x_train]
x_val = [[w2i['.start']] + x + [w2i['.end']] for x in x_val]

# sort x and y by length
x_train_sorted = sorted(x_train, key=len, reverse=True)
y_train_sorted = [y_train[x_train.index(x)] for x in x_train_sorted]
x_val_sorted = sorted(x_val, key=len, reverse=True)
y_val_sorted = [y_val[x_val.index(x)] for x in x_val_sorted]

# convert datapoints to tensor individually
x_train = [torch.tensor(x, device=device) for x in x_train_sorted]
y_train = [torch.tensor(y, device=device) for y in y_train_sorted]
x_val = [torch.tensor(x, device=device) for x in x_val_sorted]
y_val = [torch.tensor(y, device=device) for y in y_val_sorted]

# get max sequence length
maxlen = max(len(x) for x in x_train)
print(f'max sequence length of train: {maxlen}')
print('average sequence length of train:', sum(len(x) for x in x_train) / len(x_train))
maxlen = max(len(x) for x in x_val)
print(f'max sequence length of val: {maxlen}')
print(f'average sequence length of val:', sum(len(x) for x in x_val) / len(x_val))

batch_size = 32


def collate_fn(batch: list):
    x, y = zip(*batch)
    x = pad_sequence(x, batch_first=True, padding_value=w2i['.pad'])
    return x, torch.tensor(y, device=device)


train_loader = torch.utils.data.DataLoader(list(zip(x_train, y_train)), batch_size=batch_size, shuffle=True,
                                           collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(list(zip(x_val, y_val)), batch_size=batch_size, shuffle=True,
                                         collate_fn=collate_fn)


class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=300, outsize=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=w2i['.pad'])
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, outsize)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)  # TODO
        x = self.relu(x)
        x = torch.max(x, dim=1)[0]
        x = self.linear2(x)
        return x


# takes 15 hours to train!!!!
class Elman(nn.Module):
    def __init__(self, vocab_size, embedding_size=300, hsize=300, outsize=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=w2i['.pad'])
        self.layer1 = nn.Linear(2 * embedding_size, hsize)  # 2* because we concat
        self.tanh = nn.Tanh()
        self.layer2 = nn.Linear(hsize, outsize)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        # Elman layer
        b, t, e = x.size()
        if hidden is None:
            hidden = torch.zeros(b, e, dtype=torch.float, device=device)
        outs = []
        for i in range(t):
            # concat hidden and x
            inp = torch.cat([x[:, i, :], hidden], dim=1)
            out = self.layer1(inp)
            hidden = self.tanh(out)
            outs.append(hidden)
        # -> back to tensor as b, t, e
        x = torch.stack(outs, dim=1)
        x = torch.max(x, dim=1)[0]
        x = self.layer2(x)
        return x, hidden


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_size=300, hsize=300, outsize=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=w2i['.pad'])
        self.RNN = nn.RNN(embedding_size, hsize, num_layers=1, batch_first=True)
        self.layer2 = nn.Linear(hsize, outsize)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.RNN(x, hidden)
        # x = x[:, -1, :]  # only take last output
        x = torch.max(x, dim=1)[0]  # works better
        x = self.layer2(x)
        return x, hidden


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size=300, hsize=300, outsize=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=w2i['.pad'])
        self.LSTM = nn.LSTM(embedding_size, hsize, num_layers=1, batch_first=True)
        self.layer2 = nn.Linear(hsize, outsize)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.LSTM(x, hidden)
        # x = x[:, -1, :]  # only take last output
        x = torch.max(x, dim=1)[0]
        x = self.layer2(x)
        return x, hidden


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, plot_points=20):
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    gradient_norms = []

    plot_interval = len(train_loader) // plot_points

    print(f'training {model.__class__.__name__}...')
    for epoch in range(n_epochs):
        running_loss = 0.0
        train_correct = 0
        print(f'\nEpoch {epoch + 1}')
        for i, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            model.train()
            optimizer.zero_grad()
            if model.__class__.__name__ in ['Elman', 'LSTM', 'RNN']:
                y_pred, _ = model(x)
            else:
                y_pred = model(x)
            # y_pred = y_pred.softmax(dim=1)
            train_correct += (torch.argmax(y_pred, dim=1) == y).sum().item()
            loss = loss_fn(y_pred, y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i % plot_interval == plot_interval - 1:

                # track gradient norms
                total_gradient_norm = 0
                for param in model.parameters():
                    total_gradient_norm += param.grad.detach().data.norm()
                    total_gradient_norm /= len(list(model.parameters()))
                gradient_norms.append(total_gradient_norm)

                # get acc of train
                acc = train_correct / plot_interval / batch_size
                train_acc.append(acc)
                train_correct = 0

                # get loss of train
                train_loss.append(running_loss / plot_interval)
                running_loss = 0.0

                # validate
                with torch.no_grad():
                    model.eval()
                    correct = 0
                    total = 0
                    val_running_loss = 0.0
                    for x, y in itertools.islice(val_loader, 0, plot_interval):
                        if model.__class__.__name__ in ['Elman', 'LSTM', 'RNN']:
                            y_pred, _ = model(x)
                        else:
                            y_pred = model(x)
                        # y_pred = y_pred.softmax(dim=1)
                        loss = loss_fn(y_pred, y)
                        val_running_loss += loss.item()
                        correct += (torch.argmax(y_pred, dim=1) == y).sum().item()
                        total += len(y)
                    val_acc.append(correct / total)
                    val_loss.append(val_running_loss / plot_interval)

    print(f'Finished Training {model.__class__.__name__}')
    print('final train acc:', train_acc[-1])
    print('final test acc:', val_acc[-1])

    plt.plot(val_acc, label='test acc')
    plt.plot(train_acc, label='train acc')
    plt.title(f'{model.__class__.__name__}: accuracy over time')
    plt.xlabel('batch x 100')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    plt.plot(val_loss, label='test loss')
    plt.plot(train_loss, label='train loss')
    plt.title(f'{model.__class__.__name__}: loss over time')
    plt.xlabel('batch x 100')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    # plt.plot(gradient_norms)
    # plt.title(f'{model.__class__.__name__}: gradient norms over time')
    # plt.xlabel('batch x 100')
    # plt.ylabel('gradient norm')
    # plt.show()


n_epochs = 4
lr = 0.0005


### MLP ###
model = MLP(len(i2w), embedding_dim=300, hidden_dim=300, outsize=2)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader)

### Elman ###
model = Elman(len(i2w), embedding_size=300, hsize=300, outsize=2)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader)

### RNN ###
model = RNN(len(i2w), embedding_size=300, hsize=300, outsize=2)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader)

### LSTM ###
model = LSTM(len(i2w), embedding_size=300, hsize=300, outsize=2)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader)

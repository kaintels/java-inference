import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Linear(1, 10)
        self.batch = nn.BatchNorm1d(10)
        self.drop = nn.Dropout1d()
        self.layer2 = nn.Linear(10, 1)

    def forward(self, x):

        x = self.layer1(x)
        x = self.batch(x)
        x = F.relu(self.drop(x))
        x = F.relu(self.layer2(x))

        return x


X = torch.FloatTensor([[1],[2],[3],[4]])
Y = torch.FloatTensor([[40],[50],[60],[70]])

learn_rate = 0.01
n_epochs = 100

model = Net()
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(lr = learn_rate, params=model.parameters())

for i in range(0, n_epochs):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = mse_loss(y_pred,Y)
    loss.backward()
    optimizer.step()

    print(loss.item())

X_test = torch.tensor([[5]], dtype=torch.float32)

model.eval()
traced_cell = torch.jit.trace(model, X_test)

print(model(X_test))
traced_cell.save('./model/model.pt')
import numpy as np
import warnings

import sklearn.neural_network

warnings.filterwarnings('ignore')


def ReLu(arr):
    return np.maximum(arr, 0)


def softmax(arr):
    e = np.exp(arr)
    return e / e.sum()


def evaluate_acc(y_test, y_pred):
    """
    Evaluates the accuracy of a model's prediction
    :param y_test: np.ndarray - the true labels
    :param y_pred: np.ndarray - the predicted labels
    :return: float - prediction accuracy
    """
    return np.sum(y_pred == y_test) / y_pred.shape[0]


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


class MLP:

    def __init__(self, num_inputs, num_outputs, num_hidden_units=[64, 64], activation_fn=ReLu):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = len(num_hidden_units)
        self.size_layers = num_hidden_units
        self.activation_fn = activation_fn

        self.weights = []
        layers = [num_inputs] + num_hidden_units + [num_outputs]

        # initializing weights and biases (bias ignored for now)
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.75)  # idk why * .01 it was in the slides

    def fit(self, x, y):
        N, D = x.shape
        y = one_hot(y, self.num_outputs)

        def gradient1(x, y, params):
            v, w = params
            q = np.dot(x, v)
            q_01 = np.where(q > 0, 1, 0)
            z = ReLu(q)  # N x M
            yh = softmax(np.dot(z, w))  # N
            dy = yh - y  # N
            dw = np.dot(z.T, dy) / N  # M x C
            dz = np.dot(dy, w.T)  # N x M
            dv = np.dot(x.T, dz * q_01) / N  # D x M
            return [dv, dw]

        learning_rate = 0.001
        max_iters = 1e4
        epsilon = 1e-8

        norms = np.array([np.inf])
        t = 1
        while np.any(norms > epsilon) and t < max_iters:
            grad = gradient1(x, y, self.weights)
            for p in range(len(self.weights)):
                if np.isnan(np.min(grad[p])):
                    t = max_iters
                    break

                self.weights[p] = self.weights[p] - (learning_rate * grad[p])
            t += 1
            norms = np.array([np.linalg.norm(g) for g in grad])
            if (t % 10) == 0:
                print(f"{t / max_iters * 100}% complete")

        return self

    def predict(self, x):
        output = x
        for weight_matrix in self.weights:
            output = self.activation_fn(np.dot(output, weight_matrix))
        return output


if __name__ == '__main__':
    import numpy as np
    import mnist_reader

    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    # Data normalization
    X_train = X_train - np.mean(X_train)
    X_train = X_train / np.std(X_train)
    X_test = X_test - np.mean(X_test)
    X_test = X_test / np.std(X_test)

    #########################################################
    ######################## MLP ############################
    #########################################################

    # model = MLP(X_train.shape[1], 10, [128])
    # model.fit(X_train, y_train)
    # y_pred = np.argmax(model.predict(X_test), axis=1)
    # print(y_pred)
    # print(evaluate_acc(y_test, y_pred))

    #########################################################
    #################### MLP - SKLEARN ######################
    #########################################################

    # model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(128, 128), activation='tanh', solver='sgd')
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # print(evaluate_acc(y_test, y_pred))

    #########################################################
    ######################## CNN ############################
    #########################################################

    from torch.utils.data import Dataset, DataLoader

    class FashionMNISTDataset(Dataset):
        '''Fashion MNIST Dataset'''

        def __init__(self, X, y, transform=None):
            """
            Args:
                csv_file (string): Path to the csv file
                transform (callable): Optional transform to apply to sample
            """

            self.X = X.reshape(-1, 1, 28, 28)  # .astype(float);
            self.Y = y

            self.transform = transform

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            item = self.X[idx]
            label = self.Y[idx]

            if self.transform:
                item = self.transform(item)

            return (item, label)


    import torch
    import torch.nn as nn
    from torch.autograd import Variable

    num_epochs = 5
    num_classes = 10
    batch_size = 100
    learning_rate = 0.001

    train_dataset = FashionMNISTDataset(X_train, y_train)
    test_dataset = FashionMNISTDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)


    class CNN(nn.Module):
        def __init__(self, num_classes=10):
            super(CNN, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=5, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=5, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.fc = nn.Linear(7 * 7 * 128, 10)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CNN(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.float())
            labels = Variable(labels)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = Variable(images.float())
            labels = Variable(labels)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

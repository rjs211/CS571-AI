# Runs the mnist experiment

from layers import *
from mlxtend.data import loadlocal_mnist
from sklearn.metrics import precision_recall_fscore_support


def tester(model, x, y_true):
    model.eval_mode()
    out, _ = model.forward(x)
    model.train_mode()
    predictions = np.argmax(out[-1][0], axis=1)
    ground_truth = np.argmax(y_true, axis=1)
    acc = np.mean(predictions == ground_truth)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='weighted')
    return acc, precision, recall, fscore


# Load training data
x_train, y_train = loadlocal_mnist(
    images_path='train-images-idx3-ubyte',
    labels_path='train-labels-idx1-ubyte')


print('Dimensions: %s x %s' % (x_train.shape[0], x_train.shape[1]))


x_train = x_train.astype('float')
y_train = y_train.astype('float')

# Load test data
x_test, y_test = loadlocal_mnist(
    images_path='t10k-images-idx3-ubyte',
    labels_path='t10k-labels-idx1-ubyte')


x_test = x_test.astype('float')
y_test = y_test.astype('float')

# Split the full training data into train and validation
sss = StratifiedShuffleSplit(n_splits=1, test_size=(1.0/6.0), random_state=42)
sss.get_n_splits(x_train, y_train)
train_idx, valid_idx = next(sss.split(x_train, y_train))


len(train_idx), len(valid_idx)


x_train, y_train, x_val, y_val = x_train[train_idx], y_train[train_idx], x_train[valid_idx], y_train[valid_idx]

# For one hot encoding the labels
def onehotencode(i): return np.eye(10)[int(i)]


y_train_one_hot = np.asarray([onehotencode(i) for i in y_train])
y_val_one_hot = np.asarray([onehotencode(i) for i in y_val])
y_test_one_hot = np.asarray([onehotencode(i) for i in y_test])


# Normalize inputs. Calculate mean and standard deviation on only training data, then apply on test too.
train_mean = np.mean(x_train)
std = np.std(x_train)
print(train_mean, std)

x_train = (x_train - train_mean) / (std)
x_val = (x_val - train_mean) / (std)
x_test = (x_test - train_mean) / (std)


print(np.mean(x_train), np.std(x_train))
print(np.mean(x_val), np.std(x_val))
print(np.mean(x_test), np.std(x_test))


# Defines model
layers_mnist = [
    FullyConnected(16, 784),
    Tanh(),
    FullyConnected(10, 16),
]
mymlp_mnist = MultiLayerPerceptron(
    layers_mnist, SoftmaxCrossEntropyLoss(), SoftmaxLayer())

best_mlp_mnist = None
best_f1 = 0.0
losses = []
for epoch in range(100):
    mymlp_mnist.train_mode()
    _, loss = mymlp_mnist.forward(x_train, y_train_one_hot)
    mymlp_mnist.backward()

    train_acc, train_prec, train_rec, train_f1 = tester(
        mymlp_mnist, x_train, y_train_one_hot)
    print("Epoch {} train_acc: {}, train_prec: {}, train_rec: {}, train_f1: {}".format(
        epoch, train_acc, train_prec, train_rec, train_f1))

    val_acc, val_prec, val_rec, val_f1 = tester(
        mymlp_mnist, x_val, y_val_one_hot)
    print("Epoch {} val_acc: {}, val_prec: {}, val_rec: {}, val_f1: {}".format(
        epoch, val_acc, val_prec, val_rec, val_f1))

    test_acc, test_prec, test_rec, test_f1 = tester(
        mymlp_mnist, x_test, y_test_one_hot)
    print("Epoch {} test_acc: {}, test_prec: {}, test_rec: {}, test_f1: {}".format(
        epoch, test_acc, test_prec, test_rec, test_f1))

    if val_f1 > best_f1:
        best_mlp_mnist = deepcopy(mymlp_mnist)
        best_f1 = val_f1
        print("Saving as best model...")
    print(mymlp_mnist.optimize(1.0))
    print('Loss', loss)
    losses.append(loss)


test_acc, test_prec, test_rec, test_f1 = tester(
    best_mlp_mnist, x_test, y_test_one_hot)
print("TEST: test_acc: {}, test_prec: {}, test_rec: {}, test_f1: {}".format(
    test_acc, test_prec, test_rec, test_f1))


mymlp_mnist.all_outputs


plt.plot(losses)
plt.savefig('mnist_single_run.png')
plt.clf()


# Print learning curves for different learning rates
lrs = [0.7, 1., 1.3]
all_losses = []

for lr in lrs:
    layers_mnist = [
        FullyConnected(16, 784),
        Tanh(),
        FullyConnected(10, 16),
    ]
    mymlp_mnist = MultiLayerPerceptron(
        layers_mnist, SoftmaxCrossEntropyLoss(), SoftmaxLayer())

    best_mlp_mnist = None
    best_f1 = 0.0
    losses = []
    for epoch in range(100):
        mymlp_mnist.train_mode()
        _, loss = mymlp_mnist.forward(x_train, y_train_one_hot)
        mymlp_mnist.backward()

        train_acc, train_prec, train_rec, train_f1 = tester(
            mymlp_mnist, x_train, y_train_one_hot)
        print("Epoch {} train_acc: {}, train_prec: {}, train_rec: {}, train_f1: {}".format(
            epoch, train_acc, train_prec, train_rec, train_f1))

        val_acc, val_prec, val_rec, val_f1 = tester(
            mymlp_mnist, x_val, y_val_one_hot)
        print("Epoch {} val_acc: {}, val_prec: {}, val_rec: {}, val_f1: {}".format(
            epoch, val_acc, val_prec, val_rec, val_f1))

        test_acc, test_prec, test_rec, test_f1 = tester(
            mymlp_mnist, x_test, y_test_one_hot)
        print("Epoch {} test_acc: {}, test_prec: {}, test_rec: {}, test_f1: {}".format(
            epoch, test_acc, test_prec, test_rec, test_f1))

        if val_f1 > best_f1:
            best_mlp_mnist = deepcopy(mymlp_mnist)
            best_f1 = val_f1
            print("Saving as best model...")
        print(mymlp_mnist.optimize(lr))
        print('Loss', loss)
        losses.append(loss)
    all_losses.append(losses)


for lr, run in zip(lrs, all_losses):
    plt.plot(run, label='lr={}'.format(lr))
    plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error')
plt.savefig('mnist_loss_vs_lr.png')
plt.clf()

# Runs the iris experiment

from layers import *
from sklearn.metrics import precision_recall_fscore_support

# Import data
iris_data = pd.read_csv('iris.data', header=None)

data_x = iris_data.values[:, : 4].astype('float')
data_y = iris_data.values[:, 4].reshape(-1, 1)

# Split data class-wise into Train and Test
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
sss.get_n_splits(data_x, data_y)


train_val_index, test_index = list(sss.split(data_x, data_y))[0]
x_train_val, x_test = data_x[train_val_index], data_x[test_index]
y_train_val, y_test = data_y[train_val_index], data_y[test_index]

# Further split training data into train and validation
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
sss.get_n_splits(x_train_val, y_train_val)

train_index, val_index = list(sss.split(x_train_val, y_train_val))[0]
x_train, x_val = x_train_val[train_index], x_train_val[val_index]
y_train, y_val = y_train_val[train_index], y_train_val[val_index]

# Print sizes to verifyt splits
print(len(x_train), len(x_val), len(x_test))
print(len(y_train), len(y_val), len(y_test))


labels2id = {name: i for i, name in enumerate(set(data_y[:, 0]))}


# Functions to onehotencode labels
def onehotencode(i): return np.eye(3)[labels2id[i]]


y_train_one_hot = np.asarray([onehotencode(i) for i in y_train[:, 0]])
y_val_one_hot = np.asarray([onehotencode(i) for i in y_val[:, 0]])
y_test_one_hot = np.asarray([onehotencode(i) for i in y_test[:, 0]])


print(x_train.shape, x_val.shape, x_test.shape)
print(y_train_one_hot.shape, y_val_one_hot.shape, y_test_one_hot.shape)

# Define the model
layers_iris = [
    FullyConnected(6, 4),
    Sigmoid(),
    FullyConnected(3, 6),
]
mymlp_iris = MultiLayerPerceptron(
    layers_iris, SoftmaxCrossEntropyLoss(), SoftmaxLayer())


print(mymlp_iris)


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


tester(mymlp_iris, x_val, y_val_one_hot)


tester(mymlp_iris, x_test, y_test_one_hot)

# Run the model for 500 epochs
best_mlp_iris = None
best_f1 = 0.0
losses = []
for epoch in range(500):
    mymlp_iris.train_mode()
    # Do a forward pass
    _, loss = mymlp_iris.forward(x_train, y_train_one_hot)
    # Do a backward pass and accumulate gradients
    mymlp_iris.backward()

    train_acc, train_prec, train_rec, train_f1 = tester(
        mymlp_iris, x_train, y_train_one_hot)
    print("Epoch {} train_acc: {}, train_prec: {}, train_rec: {}, train_f1: {}".format(
        epoch, train_acc, train_prec, train_rec, train_f1))

    # Print validation and test scores for the model
    val_acc, val_prec, val_rec, val_f1 = tester(
        mymlp_iris, x_val, y_val_one_hot)
    print("Epoch {} val_acc: {}, val_prec: {}, val_rec: {}, val_f1: {}".format(
        epoch, val_acc, val_prec, val_rec, val_f1))

    test_acc, test_prec, test_rec, test_f1 = tester(
        mymlp_iris, x_test, y_test_one_hot)
    print("Epoch {} test_acc: {}, test_prec: {}, test_rec: {}, test_f1: {}".format(
        epoch, test_acc, test_prec, test_rec, test_f1))

    # Save best model
    if val_f1 > best_f1:
        best_mlp_iris = deepcopy(mymlp_iris)
        best_f1 = val_f1
        print("Saving as best model...")
    
    # Take an optimization step, updating weights with the gradients
    print(mymlp_iris.optimize(1))

    # Print the loss in this epoch
    print('Loss', loss)
    losses.append(loss)


plt.plot(losses)
plt.savefig('iris_single_run.png')
plt.clf()


# Print final best TEST scores
test_acc, test_prec, test_rec, test_f1 = tester(
    best_mlp_iris, x_test, y_test_one_hot)
print("TEST: test_acc: {}, test_prec: {}, test_rec: {}, test_f1: {}".format(
    test_acc, test_prec, test_rec, test_f1))


# Run the experiment multiple times for varying learning rates
all_losses = []
lrs = [0.3, 1, 1.3]
for lr in lrs:
    layers_iris = [
        FullyConnected(6, 4),
        Sigmoid(),
        FullyConnected(3, 6),
    ]
    mymlp_iris = MultiLayerPerceptron(
        layers_iris, SoftmaxCrossEntropyLoss(), SoftmaxLayer())
    best_mlp_iris = None
    best_f1 = 0.0
    losses = []
    for epoch in range(500):
        mymlp_iris.train_mode()
        _, loss = mymlp_iris.forward(x_train, y_train_one_hot)
        mymlp_iris.backward()

        train_acc, train_prec, train_rec, train_f1 = tester(
            mymlp_iris, x_train, y_train_one_hot)
        print("Epoch {} train_acc: {}, train_prec: {}, train_rec: {}, train_f1: {}".format(
            epoch, train_acc, train_prec, train_rec, train_f1))

        val_acc, val_prec, val_rec, val_f1 = tester(
            mymlp_iris, x_val, y_val_one_hot)
        print("Epoch {} val_acc: {}, val_prec: {}, val_rec: {}, val_f1: {}".format(
            epoch, val_acc, val_prec, val_rec, val_f1))

        test_acc, test_prec, test_rec, test_f1 = tester(
            mymlp_iris, x_test, y_test_one_hot)
        print("Epoch {} test_acc: {}, test_prec: {}, test_rec: {}, test_f1: {}".format(
            epoch, test_acc, test_prec, test_rec, test_f1))

        if val_f1 > best_f1:
            best_mlp_iris = deepcopy(mymlp_iris)
            best_f1 = val_f1
            print("Saving as best model...")
        print(mymlp_iris.optimize(lr))
        print('Loss', loss)
        losses.append(loss)
    all_losses.append(losses)


for lr, run in zip(lrs, all_losses):
    plt.plot(run, label='lr={}'.format(lr))
    plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('iris_loss_vs_lr.png')
plt.clf()

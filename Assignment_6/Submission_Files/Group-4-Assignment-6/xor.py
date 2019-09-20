# Runs the xor experiment

from layers import *

# Defines Data (truth table)
x = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1], ]
y = [0,
     1,
     1,
     0]

input_x = np.asarray(x, dtype=np.float32)
output_y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

# Define the model
layers = [FullyConnected(2, 2),
          Sigmoid(),
          FullyConnected(1, 2)]
mymlp = MultiLayerPerceptron(layers, SigmoidMSELoss(), Sigmoid())

print(mymlp)

# Train model for 5000 epochs
mymlp.train_mode()
losses_run = []
for _ in range(5000):
    # Take forward step
    _, loss = mymlp.forward(input_x, output_y)
    # Make backward pass accumulating gradients
    mymlp.backward()
    # Take an optimization step updating the weights with gradients 
    print(mymlp.optimize(0.475))
    # Print the loss
    print('Loss', loss)
    losses_run.append(loss)

# Test the model
mymlp.eval_mode()
out, _ = mymlp.forward(input_x)

losses = []
lrs = [0.45, 0.475, 0.5]

for lr in lrs:
    layers = [FullyConnected(2, 2),
              Sigmoid(),
              FullyConnected(1, 2)]
    mymlp = MultiLayerPerceptron(layers, SigmoidMSELoss(), Sigmoid())

    mymlp.train_mode()
    losses_run = []
    for _ in range(5000):
        _, loss = mymlp.forward(input_x, output_y)
        mymlp.backward()
        print(mymlp.optimize(lr))
        print('Loss', loss)
        losses_run.append(loss)
    losses.append(losses_run)

# Print the learning curves
for lr, run in zip(lrs, losses):
    plt.plot(run, label='lr={}'.format(lr))
    plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error')
plt.savefig('xor_loss_vs_lr.png')
plt.clf()

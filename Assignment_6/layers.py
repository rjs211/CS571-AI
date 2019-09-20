# Defines all the layers, and the MultiLayerPerceptron class

import numpy as np
import pandas as pd
import random
from copy import deepcopy

from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

deb = print

class FullyConnected():
    layer_count = 0

    def __init__(self, out_features=100, in_features=100):
        '''
            Fully Connected layer
            out_features: No of Output Features
            no_of_inp_featues: No of Input Features
        '''
        self.out_features = out_features
        self.in_features = in_features
        self.weight = np.random.normal(
            loc=0.0, scale=1/np.sqrt(self.out_features), size=(self.out_features, self.in_features))
        self.bias = np.zeros((self.out_features, 1), dtype=np.float32)
        self.weight_grad = None
        self.bias_grad = None
        self.layer_id = FullyConnected.layer_count
        FullyConnected.layer_count += 1

    def __str__(self, ):
        return 'Fully Connected Layer ID = {}: (In:{}, Out:{})'.format(self.layer_id, self.in_features, self.out_features)

    def forward(self, x):
        '''
            x: Input (N, F_in) where N = No. of samples, F_in: No. of input features
            
            returns:
                Wx + B (N, F_out) where W, B are trainable weight and bias where N: No. of samples, F_out Output dimension/features.
        '''
        assert x.shape[1] == self.in_features, "Input dimension Mismatch"
        wx = np.matmul(self.weight, np.transpose(x, (1, 0)))
        wx_plus_b = np.add(wx, self.bias)
        return np.transpose(wx_plus_b, (1, 0))

    def backward(self, y_grad, y, x):
        '''
            y_grad: Gradiant of output (N, F_out)
            y: Output of layer (N, F_out)
            x: input of layer (N, F_in)
        '''
        grad_x = np.matmul(y_grad, self.weight)

        weight_grad = np.matmul(np.transpose(y_grad, (1, 0)), x)
        if self.weight_grad is not None:
          self.weight_grad += weight_grad
        else:
          self.weight_grad = weight_grad
        assert y_grad.shape == y.shape, "y_grad.shape: {}, y.shape: {}".format(
            y_grad.shape, y.shape)

        bias_grad = np.sum(np.transpose(y_grad, (1, 0)), axis=1, keepdims=True)
        assert bias_grad.shape == self.bias.shape, "bias_grad.shape: {}, self.bias.shape: {}".format(
            bias_grad.shape, self.bias.shape)
        if self.bias_grad is not None:
          self.bias_grad += bias_grad
        else:
          self.bias_grad = bias_grad
        return grad_x

    def apply_gradients(self, lr=0.001):

        self.weight -= lr * self.weight_grad
        self.bias -= lr * self.bias_grad

        convergence_criteria = np.sum(np.absolute(
            self.weight_grad)) + np.sum(np.absolute(self.bias_grad))
        self.weight_grad = None
        self.bias_grad = None
        deb(convergence_criteria)

        if convergence_criteria < 1e-3:
            return True
        else:
            return False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class SoftmaxCrossEntropyLoss():
  """
    Combines a softmax layer with the cross-entropy loss
  """
  layer_count = 0

  def __init__(self):
      self.layer_id = SoftmaxCrossEntropyLoss.layer_count
      SoftmaxCrossEntropyLoss.layer_count += 1

  def __str__(self, ):
      return 'Softmax Cross Entropy Loss Layer ID = {}'.format(self.layer_id)

  def forward(self, x, y_true):
    '''
      x: Input (N, C) where N = No. of samples, C = No. of classes
      y_true: Target (N, C), boolean values
    '''
    # Softmax
    x_stable = x - np.max(x, axis=1, keepdims=True)
    p = np.exp(x_stable) / np.sum(np.exp(x_stable), axis=1, keepdims=True)

    # Cross Entropy Loss
    loss = y_true * \
        (- x_stable + np.log(np.sum(np.exp(x_stable), axis=1, keepdims=True)))
    loss = np.mean(np.sum(loss, axis=-1))
    return p, loss

  def backward(self, y_pred, y_true):
    '''
      y_pred: (N, C)
      y_true: (N, C)
    '''
    # Using y_pred.shape[0] as a workaround for batch_size.
    return (y_pred - y_true) / y_pred.shape[0]

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def apply_gradients(self, lr=None):
    return True


class Sigmoid():
  """
  Class for implementing sigmoid layer.
  """
  layer_count = 0

  def __init__(self):
    self.layer_id = Sigmoid.layer_count
    Sigmoid.layer_count += 1

  def __str__(self, ):
      return 'Sigmoid Layer ID = {}'.format(self.layer_id)

  def forward(self, x):
    """
    Function for implemeting the sigmoid expression.

    x: Input(N, C) where N = number of samples, C = number of classes.
    """
    y = 1.0 / (1.0 + np.exp(-x))
    return y

  def backward(self, y_grad, y, x):
    """
    Function for calculating the gradients of the current sigmoid perceptron.

    y_grad: Gradient at the output.
    y: The calculated forward(x).
    """
    w_grad = y_grad * y * (1 - y)
    return w_grad

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def apply_gradients(self, lr=None):
    return True


class Tanh():
  """
  Class for implementing tanh layer.
  """
  layer_count = 0

  def __init__(self):
    self.layer_id = Tanh.layer_count
    Tanh.layer_count += 1

  def __str__(self, ):
      return 'Tanh Layer ID = {}'.format(self.layer_id)

  def forward(self, x):
    """
    Function for implemeting the tanh expression.

    x: Input(N, C) where N = number of samples, C = number of classes.
    """
    y = np.tanh(x)
    return y

  def backward(self, y_grad, y, x):
    """
    Function for calculating the gradients of the current sigmoid perceptron.

    y_grad: Gradient at the output.
    y: The calculated forward(x).
    """
    w_grad = y_grad * (1 - (y ** 2))
    return w_grad

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def apply_gradients(self, lr=None):
    return True


class SoftmaxLayer():
  """
  Class implementing a softmax layer
  """
  layer_count = 0

  def __init__(self):
    self.layer_id = SoftmaxLayer.layer_count
    SoftmaxLayer.layer_count += 1

  def __str__(self, ):
      return 'Softmax Layer ID = {}'.format(self.layer_id)

  def forward(self, x):
    '''
      x: Input (N, C) where N = No. of samples, C = No. of classes
    '''
    # Softmax
    x_stable = x - np.max(x, axis=1, keepdims=True)
    p = np.exp(x_stable) / np.sum(np.exp(x_stable), axis=1, keepdims=True)

    return p

  def backward(self, y_grad, y, x):
    '''
    y_grad: Gradient at output
    y: (N, C)
    '''
    raise NotImplementedError(
        "Backprop not implemented for SoftmaxLayer, please use SoftmaxCrossEntropyLoss instead.")

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def apply_gradients(self, lr=None):
    return True


class SigmoidCrossEntropyLoss():
  """
    Combines a sigmoid layer with the cross-entropy loss.
  """
  layer_count = 0

  def __init__(self):
    self.layer_id = SigmoidCrossEntropyLoss.layer_count
    SigmoidCrossEntropyLoss.layer_count += 1

  def __str__(self, ):
    return 'Softmax Cross Entropy Loss Layer ID = {}'.format(self.layer_id)

  def forward(self, x, y_true):
    '''
      x: Input (N, C) where N = No. of samples, C = No. of classes
      y_true: Target (N, C), boolean values
    '''
    # Sigmoid
    p = 1.0 / (1.0 + np.exp(-x))

    # Cross Entropy Loss
    x_stable = np.maximum(x, 0)
    loss = x_stable - x * y_true + np.log(1 + np.exp(np.absolute(x)))
    loss = np.mean(np.sum(loss, axis=-1))
    return p, loss

  def backward(self, y_pred, y_true):
    '''
      y_pred: (N, C)
      y_true: (N, C)
    '''
    # Using y_pred.shape[0] as a workaround for batch_size.
    return (y_pred - y_true) / y_pred.shape[0]

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def apply_gradients(self, lr=None):
    return True


class SigmoidMSELoss():
  """
    Combines a sigmoid layer with the mean squared error loss.
  """
  layer_count = 0

  def __init__(self):
    self.layer_id = SigmoidMSELoss.layer_count
    SigmoidMSELoss.layer_count += 1

  def __str__(self, ):
      return 'Sigmoid Mean Squared Error Loss Layer ID = {}'.format(self.layer_id)

  def forward(self, x, y_true):
    '''
      x: Input (N, C) where N = No. of samples, C = No. of classes
      y_true: Target (N, C), boolean values
    '''
    # Sigmoid
    p = 1.0 / (1.0 + np.exp(-x))

    # Mean Squared Error Loss
    squared_error = (y_true - p) ** 2
    assert len(squared_error.shape) == 2
    loss = (np.sum(squared_error, axis=-1)) / 2
    loss = np.mean(loss, axis=-1)
    return p, loss

  def backward(self, y_pred, y_true):
    '''
      y_pred: (N, C)
      y_true: (N, C)
    '''
    # Using y_pred.shape[0] as a workaround for batch_size.
    return -(y_true - y_pred) * y_pred * (1 - y_pred) / y_pred.shape[0]

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def apply_gradients(self, lr=None):
    return True


class MultiLayerPerceptron():
  def __init__(self, layers, loss_layer, final_layer):
    self.layers = layers
    self.loss_layer = loss_layer
    self.final_layer = final_layer
    self.all_outputs = []
    self.train = True

  def __str__(self, ):
      summary = []
      for layer in self.layers:
          summary.append(str(layer))
      if self.final_layer is not None:
          summary.append('Final Layer: ' + str(self.final_layer))
      if self.loss_layer is not None:
        summary.append('Loss Layer: ' + str(self.loss_layer))
      return ' \n'.join(summary)

  def forward(self, x, y_true=None):
    # Training mode
    if self.train:
      assert (self.loss_layer is not None) and (
          y_true is not None), "Training mode, please pass y_true and set a loss layer"
      all_outputs = [x]

      for layer in self.layers:
        print(layer, x.shape, end=', ')
        x = layer(x)
        all_outputs.append(x)
        print(x.shape)
        # deb(layer)

      pred, loss = self.loss_layer(x, y_true)
      all_outputs.append((pred, y_true))

      # deb(len(all_outputs))
      # Save outputs for multiple forward passes
      self.all_outputs.append(all_outputs)

    # Evaluation mode
    else:
      assert (self.final_layer is not None) and (
          y_true is None), "Evaluation mode, doesn't take y_true as input"
      all_outputs = [x]

      for layer in self.layers:
        x = layer(x)
        all_outputs.append(x)

      pred = self.final_layer(x)
      loss = None
      all_outputs.append((pred, y_true))

    return all_outputs, loss

  def backward(self):
      assert self.train

      for all_outputs in self.all_outputs[::-1]:
        y_grad = self.loss_layer.backward(*all_outputs[-1])
        all_outputs = all_outputs[:-1]
        for idx, layer in enumerate(self.layers[::-1]):
          # Pass the grad at the output, output of the layer, the input of the layer to backward()
          y_grad = layer.backward(
              y_grad, all_outputs[-idx-1], all_outputs[-idx-2])

      self.all_outputs = []
      return

  def train_mode(self):
      '''
      Set train mode
      '''
      self.train = True

  def eval_mode(self):
      '''
      Set eval mode
      '''
      self.train = False

  def optimize(self, learning_rate):
      '''
      Optimizes
      '''
      conv = True
      for layer in self.layers:
        conv = layer.apply_gradients(learning_rate) and conv
      return conv

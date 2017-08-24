"""Loss functions."""

import tensorflow as tf
import keras.backend as K
import numpy as np


def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    
    assert max_grad > 0.
    
    x = y_true - y_pred
    if np.isinf(max_grad):
        return .5 * K.square(x)
    
    condition = K.abs(x) < max_grad
    squared_loss = .5 * K.square(x)
    linear_loss = max_grad * (K.abs(x) - .5 * max_grad)
  
    return tf.where(condition, squared_loss, linear_loss)  # condition, true, false



def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    
    return tf.reduce_mean(huber_loss(y_true,y_pred,max_grad))

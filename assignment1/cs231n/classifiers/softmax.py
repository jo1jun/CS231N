from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    m = X.shape[0]
    for i in range(m):
      s = X[i] @ W
      s_max = np.max(s)
      e_s = np.exp(s - s_max) # overflow 방지
      p = e_s[y[i]] / np.sum(e_s)
      Li = -np.log(p)
      loss += Li

      ds = np.zeros_like(s)
      ds[y[i]] = -1 / (np.sum(e_s) * p)
      ds += e_s[y[i]] / np.square(np.sum(e_s)) / p
      dW += X[i].T.reshape(-1,1) @ (e_s * ds).reshape(1,-1)

    loss /= m
    loss += 0.5 * reg * np.sum(np.square(W))

    dW /= m
    dW += reg * W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    m = X.shape[0]

    s = X @ W
    s_max = np.max(s, 1).reshape(-1,1)
    e_s = np.exp(s - s_max)
    p = e_s[np.arange(m), y] / np.sum(e_s, 1)
    loss = np.sum(-np.log(p))
    loss /= m
    loss += 0.5 * reg * np.sum(np.square(W))

    ds = np.zeros_like(s)
    ds[np.arange(m), y] -= 1 / (np.sum(e_s, 1) * p)
    ds += (e_s[np.arange(m), y] / np.square(np.sum(e_s, 1)) / p).reshape(-1,1)
    dW = X.T @ (e_s * ds)

    dW /= m
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

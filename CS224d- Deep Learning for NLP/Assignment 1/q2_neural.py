#This neural network has one sigmoid hidden layer and one softmax output layer

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))   #size:10*5
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))        #size:1*5
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))  #size:5*10
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))      #size:1*10

    #data size: 20*10, one row is one group of data. 20 groups in total. each group has 10 data

    ### YOUR CODE HERE: forward propagation
    h = sigmoid(np.add(np.matmul(data,W1),b1))        #data*W1 size: 20*5; h size:20*5
    y_hat = softmax(np.add(np.matmul(h,W2),b2))        #h*W2 size: 20*10; y_hat size:20*10
    cost = -np.sum(np.multiply(labels,np.log(y_hat)))
    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    gradtheta2 = np.subtract(y_hat, labels)   #gradtheta2 size: 20*10; labels size:20*10; y_hat size:20*10
    gradb2 = np.matmul(np.ones((1,20)),np.subtract(y_hat, labels))  #gradb2 size: 1*10
    gradW2 = np.matmul(np.transpose(h), gradtheta2)          #gradW2 size: 5*10
    gradh = np.matmul(gradtheta2, W2.T)             #gradh size: 20*5
    gradtheta1 = np.multiply(gradh, sigmoid_grad(h))  #gradtheta1 size: 20*5
    gradb1 = np.matmul(np.ones((1,20)), np.multiply(gradh, sigmoid_grad(h)))    #gradb1 size: 1*5
    gradW1 = np.matmul(data.T, gradtheta1)         #gradW1 size: 10*5

    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print ("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print ("Running your sanity checks...")
    ### YOUR CODE HERE

    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
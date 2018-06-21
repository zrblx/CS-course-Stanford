import numpy as np
import random

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    You should also make sure that your code works for one
    dimensional inputs (treat the vector as a row), you might find
    it helpful for your later problems.

    You must implement the optimization in problem 1(a) of the 
    written assignment!
    """

    ### YOUR CODE HERE

    try :
        col = x.shape[1]
        row = x.shape[0]
        # prevent overflow
        x = np.subtract(x, np.amax(x, axis=1).reshape(row, 1))
    except IndexError :
        #consider special case when the input x is a vector instead of a matrix
        row = 1
        x = np.subtract(x, np.amax(x))

    e_x = np.exp(x)

    if row > 1:
        div = np.sum(e_x, axis = 1)
        #reshape to make it possible for broadcasting 
        div = div.reshape(row,1)   
    else:
        div = np.sum(e_x)
    '''
    print(np.shape(e_x))
    print(e_x)
    print(np.shape(div))
    print(div)
    '''
    return np.divide(e_x,div)

def test_softmax_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print ("Running basic tests...")
    test1 = softmax(np.array([1,2]))
    print("test1 = :",test1)
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print ("test2 = :",test2)
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]))
    print ("test3 = :", test3)
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

    print ("You should verify these results!\n")

def test_softmax():
    """ 
    Use this space to test your softmax implementation by running:
        python q1_softmax.py 
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print ("Running your tests...")
    ### YOUR CODE HERE

    ### END YOUR CODE  

if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()

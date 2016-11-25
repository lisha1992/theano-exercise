# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 09:19:58 2016

@author: ceciliaLee
"""

import cPickle, gzip
import numpy as np
import theano
import theano.tensor as T
# Load the dataset
f = gzip.open('/Users/CeciliaLee/Desktop/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables
    
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    
    return shared_x, T.cast(shared_y, 'int32') 
    # during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn’t make sense) therefore instead of returning
    # ‘‘shared_y‘‘ we will have to cast it to int. 
    
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    
    batch_size=500 # size of the minibatch
    
    # accessing the third minibatch of the training set
    data  = train_set_x[2 * batch_size: 3 * batch_size]
    label = train_set_y[2 * batch_size: 3 * batch_size]
    

## 0-1 loss function
    # zero_one_loss is a Theano variable representing a symbolic
    # expression of the zero one loss ; to get the actual value this
    # symbolic expression has to be compiled into a Theano function (see
    # the Theano tutorial for more details)
    zero_one_loss = T.sum(T.neq(T.argmax(p_y_given_x), y))
  ## T.neq(): Returns a variable representing the result of logical inequality (a!=b).
    
    
## Negative Log-Likelihood Loss

    NLL = -T.sum(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
    # note on syntax: T.arange(y.shape[0]) is a vector of integers [0,1,2,...,len(y)].
    # Indexing a matrix M by the two vectors [0,1,...,K], [a,b,...,k] returns the
    # elements M[0,a], M[1,b], ..., M[K,k] as a vector.  Here, we use this
    # syntax to retrieve the log-probability of the correct labels, y.
    

## Stochastic Gradient Descent
# Minibatch SGD
# assume loss is a symbolic description of the loss function given
# the symbolic variables params (shared variable), x_batch, y_batch;

    # compute gradient of loss with respect to params
    d_loss_wrt_params = T.grad(loss, params)
    # compile the MSGD step into a theano function
    updates = [(params, params - learning_rate * d_loss_wrt_params)]
    MSGD = theano.function([x_batch,y_batch], loss, updates=updates)
    
    for (x_batch, y_batch) in train_batches:
        # here x_batch and y_batch are elements of train_batches and
        # therefore numpy arrays; 
        # function MSGD also updates the params print(’Current loss is ’, MSGD(x_batch, y_batch))
        if stopping_condition_is_met:
            return params
            
## Regularization, avoid overfitting
    # symbolic Theano variable that represents the L1 regularization term
    L1 = T.sum(abs(param))
    # symbolic Theano variable that represents the squared L2 term
    L2_sqr = T.sum(param ** 2)
    # the loss
    loss = NLL + lambda_1 * L1 + lambda_2 * L2
    
    
## Early-stopping, avoid overfitting
    #early-stopping parameters,    
    patience = 5000 # look as this many examples regardless
    patience_increase = 2 # wait this much longer when a new best is found
    improvement_threshold = 0.995 # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience/2) 
        # go through this many minibatches before checking the network
        # on the validation set; in this case we check every epoch
    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()
    done_looping = False
    epoch = 0
    
    while (epoch < n_epochs) and (not done_looping):
        # Report "1" for first epoch, "n_epochs" for last epoch
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            
            d_loss_wrt_params = ... # compute gradient
            params -= learning_rate * d_loss_wrt_params # gradient descent
            
            # iteration number. We want it to start at 0.
            iter = (epoch - 1) * n_train_batches + minibatch_index
            # note that if we do ‘iter % validation_frequency‘ it will be # true for iter = 0 which we do not want. We want it true for # iter = validation_frequency - 1.
            if (iter + 1) % validation_frequency == 0:
                this_validation_loss = ... # compute zero-one loss on validation set if this_validation_loss < best_validation_loss:
                # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                    patience = max(patience, iter * patience_increase)
                best_params = copy.deepcopy(params)
                best_validation_loss = this_validation_loss
            if patience <= iter:
                done_looping = True
                break
     # POSTCONDITION:
            # best_params refers to the best out-of-sample parameters observed during the optimization
            
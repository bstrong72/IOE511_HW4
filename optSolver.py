# IOE 511/MATH 562, University of Michigan
# Code written by: Jiale Zha

# Run a chosen optiomization algorithm on a chosen ML problem

import numpy as np
import functions
import algorithms 

def optSolverML_with_tracker(problem, method, options):
    # Function that runs a chosen algorithm on a chosen problem and tracks
    # behaviour of target function and step size
    #
    #           Inputs: problem, method, options (classes)
    #           Outputs: final weight (w), 
    #                    training loss over epochs (loss_train_trace),
    #                    training accuracy over epochs (acc_train_trace),
    #                    test loss over epochs (loss_test_trace),
    #                    test accuracy over epochs (acc_test_trace) 


    # load dataset and initial weight
    x = problem.x0
    X_train = problem.X_train
    y_train = problem.y_train.reshape(-1)
    X_test = problem.X_test
    y_test = problem.y_test.reshape(-1)
    n = len(y_train) + len(y_test) # total sample size

    # set weight and step size update functions and number of iterations per epoch
    if method.name == 'GradientDescent':
        # set weight update functions: Gradient Descent
        weight_update_func = algorithms.GDStep

        # set constant step size
        step_size_update_func = lambda alpha_bar, k: alpha_bar

        # one gradient evaluation per epoch
        num_steps = 1

    elif method.name == 'StochasticGradient':
        # set weight update functions: Stochastic Gradient Descent
        weight_update_func = algorithms.SGDStep

        # number of updates depend on batch size
        num_steps = int(n / method.options.batch_size) + 1
        
        if method.options.step_type == 'Constant':
            # set constant step size
            step_size_update_func = lambda alpha_bar, k: alpha_bar

        elif  method.options.step_type == 'Diminishing':
            # set diminishing step size
            step_size_update_func = lambda alpha_bar, k: alpha_bar/k

        else:
            print('Warning: Step size not specified yet!')
    else:
        print('Warning: Optimization method not implemented yet!')

    # trackers for loss function value and step size
    alpha_bar = method.options.alpha_bar
    loss_train_trace = []
    loss_test_trace = []
    acc_train_trace = []
    acc_test_trace = []
    
    # initial loss and gradient
    loss_f = problem.compute_f(x, X_train, y_train)
    loss_g = problem.compute_g(x, X_train, y_train)

    for epoch in range(options.num_epoch):
        for k in range(num_steps):
            # update step size
            alpha_k = step_size_update_func(alpha_bar, epoch * num_steps + k + 1)

            x_new, loss_f_new, loss_g_new, d, alpha_k = weight_update_func(
                x, loss_f, loss_g, X_train, y_train, alpha_k, 
                problem, method, options)
        
            # update weight, loss and gradient
            x = x_new
            loss_f = loss_f_new
            loss_g = loss_g_new

        # record current loss and accuracy
        loss_train_trace.append(problem.compute_f(x, X_train, y_train))
        loss_test_trace.append(problem.compute_f(x, X_test, y_test))
        acc_train_trace.append(np.mean(problem.pred_func(x, X_train) == y_train))
        acc_test_trace.append(np.mean(problem.pred_func(x, X_test) == y_test))
 
    return (x, 
            np.array(loss_train_trace), 
            np.array(acc_train_trace), 
            np.array(loss_test_trace),
            np.array(acc_test_trace)
            )


def optSolverML(problem,method,options):
    # Function that runs a chosen algorithm on a chosen ML problem
    #           Inputs: problem, method, options (classes)
    #           Outputs: final iterate (x), final training loss and accuracy
    #                    final test loss and accuracy

    (x, loss_train_trace, acc_train_trace, 
     loss_test_trace, acc_test_trace) = optSolverML_with_tracker(
        problem, method, options)
    return (x, 
            loss_train_trace[-1], 
            acc_train_trace[-1], 
            loss_test_trace[-1], 
            acc_test_trace[-1]
            )
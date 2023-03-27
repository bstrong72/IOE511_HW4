# IOE 511/MATH 562, University of Michigan
# Code written by: Jiale Zha

# Compute the next step for all iterative optimization algorithms given current solution x:
# (1) Gradient Descent with backtracking line search
# (2) Stochastic Gradient with specified step size

import numpy as np
import collections

def BacktrackLineSearch(w, d, loss_f, loss_g, X, y, alpha_bar, 
                        problem, method, options):
    # Function that: (1) updates the iterate; and, 
    #                (2) computes the function and gradient at the new iterate
    # 
    #           Inputs: w, d, loss_f, loss_g, X, y, alpha_bar, problem, method, options
    #           Outputs: w_new, loss_f_new, loss_g_new, d, alpha

    # initial value of step size, w and loss
    alpha = alpha_bar
    w_new = w + alpha * d
    loss_f_new = problem.compute_f(w_new, X, y)

    # descent value threshold
    loss_f_descent = method.options.c1 * loss_g.T@d
    while loss_f_new > loss_f + alpha * loss_f_descent:
        # update step size
        alpha = method.options.tau * alpha

        # update w and f
        w_new = w + alpha * d
        loss_f_new = problem.compute_f(w_new, X, y)

    # update gradient 
    loss_g_new = problem.compute_g(w_new, X, y)
    return w_new, loss_f_new, loss_g_new, d, alpha

def GDStep(w, loss_f, loss_g, X, y, alpha_bar, problem, method, options):
    # Function that: (1) computes the GD step; (2) updates the iterate; and, 
    #                (3) computes the function and gradient at the new iterate
    # 
    #           Inputs: w, loss_f, loss_g, X, y, alpha_bar, problem, method, options
    #           Outputs: w_new, loss_f_new, loss_g_new, d, alpha

    # search direction is -g
    d = -loss_g

    # Backtracking Line Search
    return BacktrackLineSearch(w, d, loss_f, loss_g, X, y, alpha_bar, 
                               problem, method, options)

def SGDStep(w, loss_f, loss_g, X, y, alpha, problem, method, options):
    # Function that: (1) computes the SGD step; and
    #                (2) updates the iterate
    # 
    #           Inputs: w, loss_f, loss_g, X, y, alpha, problem, method, options
    #           Outputs: w_new, loss_f_new(None), loss_g_new(None), d, alpha

    # draw random sample
    sample_idx = np.random.randint(0, len(X), method.options.batch_size)
    X_sample = X[sample_idx]
    y_sample = y[sample_idx]

    # search direction is -g
    loss_g = problem.compute_g(w, X_sample, y_sample)
    d = -loss_g

    # update weight
    w_new = w + alpha * d
    loss_f_new = None
    loss_g_new = None
    return w_new, loss_f_new, loss_g_new, d, alpha
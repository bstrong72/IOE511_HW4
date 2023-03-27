# Compute the next step for all iterative optimization algorithms given current solution x:
# (1) Gradient Descent
# (2) Stochastic Gradient

import functions as fnc
import numpy as np


def GDStep(w, f, g, X, y, problem, method, options):
    # Set the search direction d to be -g
    d = -g
    X = problem.X
    y = problem.y
    # determine step size
    if method.step_type == 'Constant':
        # set alphas and compute new x based on step size
        alpha = method.step_size
        w_new = w + alpha * d

        # Compute new values of f, g , H
        f_new = problem.compute_f(w_new, X, y)
        g_new = problem.compute_g(w_new, X, y)

    elif method.step_type == 'Backtracking':
        # initialize constants for backtracking subroutine
        alpha = method.step_size
        c1 = 1e-4
        Tau = 0.5

        # search for new alpha until subroutine conditions are met
        while problem.compute_f(w + (alpha * d), X, y) > (f + (c1 * alpha * np.matmul(g, np.transpose(d)))):
            alpha = Tau * alpha
        # compute new x based on step size
        w_new = w + alpha * d
        # Compute new values of f, g , H
        f_new = problem.compute_f(w_new, X, y)
        g_new = problem.compute_g(w_new, X, y)
    else:
        print('Warning: step type is not defined')

    return w_new, f_new, g_new, d, alpha


def SGDStep(w, loss_f, loss_g, X, y, problem, method, options):
    # Function that: (1) computes the SGD step; and
    #                (2) updates the iterate
    # 
    #           Inputs: w, loss_f, loss_g, X, y, problem, method, options
    #           Outputs: w_new, loss_f_new(None), loss_g_new(None), d, alpha

    alpha = method.step_size

    # draw random sample
    sample_idx = np.random.randint(0, len(X), options.batch_size)
    X_sample = X[sample_idx]
    y_sample = y[sample_idx]
    loss_g = problem.compute_g(w, X_sample, y_sample)

    # search direction is -g
    d = -loss_g
    w_new = w + alpha * d
    loss_f_new = problem.compute_f(w_new, X, y)
    loss_g_new = None
    return w_new, loss_f_new, loss_g_new, d, alpha


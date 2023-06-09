# Compute the next step for all iterative optimization algorithms given current solution x:
# (1) Gradient Descent

import functions as fnc
import numpy as np


def GDStep(x, f, g, problem, method, options):
    # Set the search direction d to be -g
    d = -g

    # determine step size
    if method.step_type == 'Constant':
        # set alphas and compute new x based on step size
        alpha = method.constant_step_size
        x_new = x + alpha * d

        # Compute new values of f, g , H
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)

    elif method.step_type == 'Backtracking':
        # initialize constants for backtracking subroutine
        alpha = 1
        c1 = 1e-4
        Tau = 0.5

        # search for new alpha until subroutine conditions are met
        while problem.compute_f(x + (alpha * d)) > (f + (c1 * alpha * np.matmul(g, np.transpose(d)))):
            alpha = Tau * alpha
        # compute new x based on step size
        x_new = x + alpha * d
        # Compute new values of f, g , H
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)
    else:
        print('Warning: step type is not defined')

    return x_new, f_new, g_new, d, alpha


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
    loss_g = problem.compute_g(w, X_sample, y_sample)

    # search direction is -g
    d = -loss_g
    w_new = w + alpha * d
    loss_f_new = None
    loss_g_new = None
    return w_new, loss_f_new, loss_g_new, d, alpha

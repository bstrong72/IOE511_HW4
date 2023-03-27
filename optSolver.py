# IOE 511/MATH 562, University of Michigan
# Code written by: Albert S. Berahas & Jiahao Shi

# Function that runs a chosen algorithm on a chosen problem
#           Inputs: problem, method, options (structs)
#           Outputs: final iterate (x) and final function value (f)

import numpy as np
import functions as fnc
import algorithms as alg


def optSolver_Strong_Benjamin(problem, method, options):
    # compute initial function/gradient/Hessian or inverse hessian approximation
    w = problem.w0
    X = problem.X
    y = problem.y
    f = problem.compute_f(w, X, y)
    g = problem.compute_g(w, X, y)

    # s_list = []
    # y_list = []
    # Compute initial norm, set norm of x0
    norm_g = np.linalg.norm(g, ord=np.inf)
    norm_g_w0 = np.linalg.norm(g, ord=np.inf)

    # set initial iteration counter
    k = 0
    # code for plotting ks and alphas
    k_list = []
    # initialize y values for plot
    plot_vals = []
    # alpha_list = []
    plot_vals.append(f)

    # while termination conditions are not met perform this loop
    while k < options.max_iterations:
        # Compute new x, g, H, f based on the selected method and options
        if method.name == 'GradientDescent':
            w_new, f_new, g_new, d, alpha = alg.GDStep(w, f, g, X, y, problem, method, options)
        elif method.name == 'StochasticGradient':
            w_new, f_new, g_new, d, alpha = alg.SGDStep(w, f, g, X, y, problem, method, options)
        else:
            print('Warning: method is not implemented yet')

        # update old and new function values        
        """x_old = x
        f_old = f
        g_old = g
        norm_g_old = norm_g"""

        # Assign new values into corresponding variables
        w = w_new
        f = f_new
        g = g_new

        # increment iteration counter
        k_list.append(k)
        plot_vals.append(f)

        # update sk and yk update counter (may not be the same as iterations
        k += 1

        # print(list(y_list))
        # print(list(s_list))
        # print(options.term_tol*np.linalg.norm(s_list[-1], ord=2)*np.linalg.norm(y_list[-1], ord=2))

        # alpha_list.append(alpha)
        # update norm for termination condition use
        norm_g = np.linalg.norm(g, ord=np.inf)
        # uncomment these to get values for plots
    return w, f, k_list, plot_vals #, alpha_list

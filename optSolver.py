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
    x = problem.x0
    f = problem.compute_f(x)
    g = problem.compute_g(x)

    # s_list = []
    # y_list = []
    # Compute initial norm, set norm of x0
    norm_g = np.linalg.norm(g, ord=np.inf)
    norm_g_x0 = np.linalg.norm(g, ord=np.inf)

    # set initial iteration counter
    k = 0
    # code for plotting ks and alphas
    k_list = []
    # initialize y values for plot
    plot_vals = []
    # alpha_list = []

    # Set x star based on problem
    """if problem.name == 'Quadratic2':
        f_star = -1.02685110
    elif problem.name == 'Quadratic10':
        f_star = -11.7334285
    elif problem.name == 'Rosenbrock':
        f_star = 0
    else:
        print('Warning: function is not implemented yet')"""


    # while termination conditions are not met perform this loop
    while norm_g > options.term_tol * max(norm_g_x0, 1) and k < options.max_iterations:
        #print(norm_g)
        #print(options.term_tol * max(norm_g_x0, 1))
        # Compute new x, g, H, f based on the selected method and options
        if method.name == 'GradientDescent':
            x_new, f_new, g_new, H_new, d, alpha = alg.GDStep(x, f, g, H, problem, method, options)
        #elif method.name == 'Newton':
        #    x_new, f_new, g_new, H_new, d, alpha = alg.Newton_Step(x, f, g, H, problem, method, options)
        #elif method.name == 'Mod_Newton':
        #    x_new, f_new, g_new, H_new, d, alpha = alg.Mod_Newton_Step(x, f, g, H, problem, method, options)
        #elif method.name == 'BFGS':
        #    x_new, f_new, g_new, H_new, d, alpha = alg.BFGS_Step(x, f, g, H_inv, problem, method, options)
        #elif method.name == 'LBFGS':
        #    x_new, f_new, g_new, d, alpha, s_k, y_k = alg.L_BFGS_Step(x, f, g, H_inv, problem, method, options, k, s_list, y_list)
        else:
            print('Warning: method is not implemented yet')

        # update old and new function values        
        """x_old = x
        f_old = f
        g_old = g
        norm_g_old = norm_g"""

        # increment iteration counter
        # k_list.append(k)
        # plot_vals.append(f)

        # Assign new values into corresponding variables
        x = x_new
        f = f_new
        g = g_new
        if method.name != 'LBFGS':
            H = H_new
            H_inv = H_new



        # update sk and yk update counter (may not be the same as iterations
        k += 1

        #print(list(y_list))
        #print(list(s_list))
        #print(options.term_tol*np.linalg.norm(s_list[-1], ord=2)*np.linalg.norm(y_list[-1], ord=2))

        # alpha_list.append(alpha)
        # update norm for termination condition use
        norm_g = np.linalg.norm(g, ord=np.inf)
                # uncomment these to get values for plots
    return x, f, # k_list, plot_vals, alpha_list

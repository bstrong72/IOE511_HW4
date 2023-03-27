# IOE 511/MATH 562, University of Michigan
# Code written by: Albert S. Berahas & Jiahao Shi

# Function that runs a chosen algorithm on a chosen problem
#           Inputs: problem, method, options (structs)
#           Outputs: final iterate (x) and final function value (f)

import numpy as np
import functions
import algorithms 

def optSolver_Fitkin_Graham(problem,method,options):
#     x = problem.x0
#     f = problem.compute_f(X,y)
#     g = problem.compute_g(X,y)  
#     H = problem.compute_H(x)
#     norm_g = np.linalg.norm(g, ord = np.inf)
    w = problem.w0
    X = problem.X
    y = problem.y
    wk=[]
    fk=[]
    loss_f = problem.compute_f(w, X, y)
    loss_g = problem.compute_g(w, X, y)
    alpha = method.constant_step_size
#     g0 = g
    # set initial iteration counter
    k = 0
    count = 0
    Flag = True
    while k < options.max_iterations:
#         print("max_iterations = ", options.max_iterations)
            #   and  norm_g >= options.term_tol*max(np.linalg.norm(g0, ord=np.inf), 1):
        
#         if  (options.max_iterations<=k):
#             Flag=False
#             return x,f,xk,fk, k, count
#         elif (norm_g <= options.term_tol*max(np.linalg.norm(problem.compute_g(problem.x0), np.inf), 1)):
#             Flag=False
#             return x, f, xk, fk, k, count
        
        if method.name == 'GradientDescent':
            x_new,f_new,g_new, d, alpha = algorithms.gradient_descent_backtracking(f, g, x, problem, method, options, alpha=1, tau=0.5, c1= 1e-4, max_iter=1000, eps=1e-6)
        elif method.name == 'Newton': 
            x_new,f_new,g_new, d, alpha, count = algorithms.modified_newtons_method(f, g, H, x, problem, method, options, max_iter=1000, tol=1e-6)
        elif method.name == 'BFGS': 
            x_new,f_new,g_new, d, alpha, count = algorithms.bfgs(f, g, x, problem, method, options, eps=1e-6, max_iter=1000)
        elif method.name == 'L-BFGS_2': 
            x_new,f_new,g_new, count = algorithms.L_BFGS(f, g, x, problem, method, options, m=2, k = k, max_iter=1000, epsilon=1e-6)
        elif method.name == 'L-BFGS_5': 
            x_new,f_new,g_new, count = algorithms.L_BFGS(f, g, x, problem, method, options, m=5, k = k, max_iter=1000, epsilon=1e-6)
        elif method.name == 'L-BFGS_10': 
            x_new,f_new,g_new, count = algorithms.L_BFGS(f, g, x, problem, method, options, m=10, k = k, max_iter=1000, epsilon=1e-6)
        elif method.name == 'GDStep':
            w_new, loss_f_new, loss_g_new, d, alpha = algorithms.GDStep(w, loss_f, loss_g, X, y, problem, method, options)
        elif method.name == 'StochasticGradient':
            w_new, loss_f_new, loss_g_new, d, alpha = algorithms.SGStep(w, loss_f, loss_g, X, y, alpha, problem, method, options, k = k)

    
        else:
            print('Warning: method is not implemented yet')
    
        # update old and new function values        
        if k==0:
            wk.append(w)
            fk.append(loss_f)
        w_old = w; f_old = loss_f; g_old = loss_g; #norm_g_old = norm_g
        w = w_new; loss_f = loss_f_new; loss_g = loss_g_new; #norm_g = np.linalg.norm(g, ord= np.inf)

        wk.append(w); fk.append(loss_f)
        # increment iteration counter
        k +=  options.batch_size
                
    return w,loss_f,wk,fk, k, count
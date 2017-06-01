import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import GPyOpt

def f(x):
    """
    Function that will produce 'real' evaluations
    for our X values.

    f is just the L2 norm squared. something simple.
    """

    x = np.atleast_2d(x)
    return (np.sum(x**2, axis=1))[:,None]

def rastrigin(x):
    """
    Function that is more complicated to optimize than
    a simple thing like the L2 norm.
    """
    x = np.atleast_2d(x)
    d = x.shape[1]
    A = 10
    return A*d + np.sum(x**2-A*np.cos(10*np.pi*x), axis=1)[:,None]

def add_suggested_sample(myProblem, X_subset, Y_subset, evaluation_func):
    """
    Given GPyOpt fitted problem, fetches the next suggested sample,
    finds its evaluation, and adds to X_subset
    TODO: pass evaluation func as a parameter instead of hardcoding
    """
    x_new = myProblem.suggested_sample

    y_new = evaluation_func(x_new)

    X_subset = np.vstack((X_subset, x_new))

    Y_subset = np.vstack((Y_subset, y_new))

    return (X_subset, Y_subset)


def set_params(n, num_init_samples, evaluation_func):
    # Obtain coordinates and function
    X_all = np.random.randn(n, 128)
    
    # precompute y-values we will actually use
    Y_all = evaluation_func(X_all)
    
    # subset X and Y (so our function can actually suggest new values)
    X_subset = X_all[:num_init_samples,:]
    Y_subset = Y_all[:num_init_samples:]
    
    # set the domain appropriately
    # must be larger than the X subset, so it has points it can sample
    domain = [{'name':'whocares', 'type':'bandit', 'domain': X_all}]

    return (X_all, Y_all, X_subset, Y_subset, domain)

def find_global_minimum(Y_all):
    """
    does what it says.
    """
    return np.min(Y_all)


def run(n, num_init_samples, evaluation_func, tolerance = 0.000001):
    """
    Returns # trials to converge gaussian proc, 
    and # for random guessing

    Stop when we hit the "true minimum" (so we don't run so effing much)
    """

    (X_all, Y_all, X_subset, Y_subset, domain) = set_params(n, num_init_samples, evaluation_func)

    # loop adding to subset
    # at 50 samples it hits the limit of the domain
    # (which is set to be the 100 points of X_all)
    
    x_opt = []
    fx_opt = []

    globalmin = find_global_minimum(Y_all)
    first_min_gpy = 0
    for i in range(n-num_init_samples): 
        myProblem = GPyOpt.methods.BayesianOptimization(f = None, 
                                                    X = X_subset, 
                                                    Y = Y_subset, 
                                                    domain=domain,
                                                    acquisition_type='MPI')
    
        fx_opt = myProblem.fx_opt[0]

        if np.abs(fx_opt-globalmin)<tolerance:
            first_min_gpy = i
            print 'hit true min at iter={0}'.format(i)
            break
    
        (X_subset, Y_subset) = add_suggested_sample(myProblem, X_subset, Y_subset, evaluation_func)
    
    first_min_rand = 0

    # how long does it take to random-sample your way to global min?
    # this is a simple expectation but w/e, we can numerically calculate
    for i,p in enumerate(np.random.permutation(Y_all)):
        if np.abs(p-globalmin)<tolerance:
            first_min_rand = i
            break

    return (first_min_gpy, first_min_rand)    

def run_experiments(max_n, step, trials_per_n, evaluation_func):
    trial_points = np.arange(start=step, stop=max_n, step=step)

    average_improvements = []
    for n in trial_points:
        improvements = []
        for i in range(trials_per_n):
            print 'iter {0}'.format(i)
            improvements.append(run(n, 2, evaluation_func))
   
        average_improvement = np.mean([i[1]-i[0] for i in improvements])
        print 'average_improvement at {0}: {1}'.format(n, average_improvement)
        average_improvements.append(average_improvement)

    plt.plot(trial_points, average_improvements)
    plt.show()


run_experiments(max_n=1000, step=900, trials_per_n=25, evaluation_func=rastrigin)

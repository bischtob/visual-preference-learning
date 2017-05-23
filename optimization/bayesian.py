import GPyOpt

from sklearn.base import BaseEstimator

class BUCB(BaseEstimator):
    def __init__(self, domain, n_iter=10, ac_fun='MPI', verbosity=True):
        self.n_iter = n_iter # number of iterations
        self.domain = [{'name':'prompt', 'type':'bandit', 'domain':domain}] # domain to optimize over
        self.ac_fun = ac_fun # aquisition function
    	self.model_type = ‘GP’ # the function class to use as prior
    	self.initial_design_numdata = 0 # initial evaluations before optimization
    	self.evaluator_type = ‘sequential’ # optimization type
    	self.verbosity = verbosity # write status

    def get_params(self, deep=True):
        return {"n_iter": self.n_iter, “domain”: self.domain, "model":self.ac_fun, "model_type":self.model_type, "initial_design_numdata":self.initial_design_numdata, "evaluator_type":self.evaluator_type, "verbosity":self.verbosity}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.parameter = value
        return self

    def optimize(self, func):
        # Functions handles the actual GP optimization via GPyOpt
        # Set up the model
        self.model = GPyOpt.methods.BayesianOptimization(f=func,
                                                         domain=self.domain, 
                                                         initial_design_numdata=self.initial_design_numdata,
                                                         acquisition_type=self.ac_func,
                                                         exact_feval=False,
                                                         normalize_Y=False)

        # Run optimization (this should leads to repeated calls to the front-end)
        self.model.run_optimization(self.n_iter)
    
        return self

if __name__ == "__main__":
    pass

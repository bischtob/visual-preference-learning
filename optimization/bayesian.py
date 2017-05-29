import GPyOpt
import numpy as np

from sklearn.base import BaseEstimator

class BOE(BaseEstimator):
    def __init__(self, x_init, y_init, domain,  T_init=2.0, ac_func='MPI'):
        self.domain = [{'name':'prompt', 'type':'bandit', 'domain':domain}] # domain to optimize over
        self.ac_func = ac_func # aquisition function
        self._X = x_init
        self._Y = y_init
        self._T = T_init

    def get_params(self, deep=True):
        return {"domain": self.domain, "model":self.ac_func}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.parameter = value
        return self

    def next(self):
        # Functions handles the actual GP optimization via GPyOpt
        # Evaluate the model with simulated annealing
        if np.random.rand() < np.exp(-self._T):
            self.model = GPyOpt.methods.BayesianOptimization(f = None, 
                                                             X = self._X, 
                                                             Y = self._Y,
                                                             domain=self.domain,
                                                             acquisition_type=self.ac_func)
            return self.model.suggested_sample
        else: 
            # Update the annealing temperature and return random sample
            self._T = 0.65*self._T
            ind = np.random.randint(0, self.domain[0]['domain'].shape[0])
            return self.domain[0]['domain'][ind,:] 

    def update_history(self, x, y):
        self._X = np.vstack((self._X, x))
        self._Y = np.vstack((self._Y, y))

if __name__ == "__main__":
    pass

import GPyOpt
import numpy as np

from sklearn.base import BaseEstimator

class BOE(BaseEstimator):
    def __init__(self, domain,  ac_func='MPI'):
        self.domain = [{'name':'prompt', 'type':'bandit', 'domain':domain}] # domain to optimize over
        self.ac_func = ac_func # aquisition function
        self._X = None
        self._Y = None
        self.next_sample = None

    def get_params(self, deep=True):
        return { "domain": self.domain, "model":self.ac_func}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.parameter = value
        return self

    def optimize(self, x, y):
        # Functions handles the actual GP optimization via GPyOpt
        # Update the sampling history
        self._update_history(x,y)

        # Evaluate the model
        self.model = GPyOpt.methods.BayesianOptimization(f = lambda x: np.ones((x.shape[0],1)), 
                                                         X = self._X, 
                                                         Y = self._Y,
                                                         normalize_Y=False,
                                                         domain=self.domain,
                                                         acquisition_type=self.ac_func)

        # Update the next sample to evaluate model on
        self.next_sample = self.model.suggested_sample[-1,:]

    def _update_history(self, x, y):
        if self._X is not None and self._Y is not None:
            self._X = np.vstack([self._X, x])
            self._Y = np.vstack([self._Y, y])
        else:
            self._X = x
            self._Y = y

if __name__ == "__main__":
    pass

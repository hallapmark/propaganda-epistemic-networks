from numpy import random
from agents.bayesianupdaters.bayesianbinomialupdater import BayesianBinomialUpdater
from agents.crsupervisor import CredenceBasedSupervisor

""" A scientist who runs experiments on a binomial distribution, and who stops 
experimenting when credence is below a certain threshold."""
class BinomialEthicalScientist(BayesianBinomialUpdater, CredenceBasedSupervisor): 
    def __init__(self, n: int, epsilon: float, stop_threshold: float, prior: float):
        print(f"Init BinomialEthicalScientist. Credence: {prior}")
        super().__init__(k = None, 
                         n = n,
                         epsilon = epsilon,
                         stop_threshold = stop_threshold,
                         prior = prior)
    
    # CredenceBasedSupervisor mandatory method implementations
    def _stop_action(self):
        print(f'Stop action ran!')
        self.k = None
    
    def _continue_action(self):
        self._experiment(self.n, self.epsilon)
        self.bayes_update_credence()
        
    # Experiment
    def _experiment(self, n: int, epsilon):
        self.k = random.binomial(n, 0.5 + epsilon)
        print(f'Experiment ran. k = {self.k}, n = {self.n}')
    


import numpy as np
from agents.bayesianupdaters.bayesianbinomialupdater import BayesianBinomialUpdater
from agents.crsupervisor import CredenceBasedSupervisor
from agents.experimenters.binomialexperimenter import BinomialExperiment
from typing import Optional

""" A scientist who runs experiments on a binomial distribution, and who stops 
experimenting when credence is below a certain threshold."""
class BinomialEthicalScientist(BayesianBinomialUpdater, CredenceBasedSupervisor): 
    def __init__(self, n_per_round: int, epsilon: float, stop_threshold: float, prior: float):
        print(f"Init BinomialEthicalScientist. Credence: {prior}")
        super().__init__(epsilon = epsilon,
                         stop_threshold = stop_threshold,
                         prior = prior)
        self.n_per_round = n_per_round
        self.binomial_experiment: Optional[BinomialExperiment] = None
    
    # CredenceBasedSupervisor mandatory method implementations
    def _stop_action(self):
        self.binomial_experiment = None
        self._finally()
    
    def _continue_action(self):
        self._experiment(self.n_per_round, self.epsilon)
        self._finally()
        
    def _finally(self):
        self.bayes_update_credence()

    # BinomialExperimenter implementation
    def get_experiment_data(self) -> Optional[BinomialExperiment]:
        return self.binomial_experiment
        
    # Experiment
    def _experiment(self, n: int, epsilon):
        k = np.random.binomial(n, 0.5 + epsilon)
        self.binomial_experiment = BinomialExperiment(k, n)
    
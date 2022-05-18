from agents.experimenters.binomialexperimenter import *
from network.network import ENetworkForBinomialUpdating
from typing import List

class SelectiveSharingPropagandist(BinomialExperimenter):
    def __init__(self, scientists: List[BinomialExperimenter]):
        self.scientists = scientists

    # BinomialExperimenter implementation
    def get_experiment_data(self) -> Optional[BinomialExperiment]:
        experiments = [scientist.get_experiment_data() for scientist in self.scientists]
        experiments = [experiment for experiment in experiments if experiment]
        if experiments:
            # Get the lowest evidence for B (i.e. highest evidence for A)
            return min(experiments, key = lambda exp: exp.k)
        else:
            return None
        # TODO: Implement safety. SelectiveSharingPropagandist cannot handle networks 
        # where n varies! (If n varies, a )
        # if not (experiments.count(experiments[0].n) == len(experiments)):
        #     raise NotImplementedError("SelectiveSharingPropagandist can currently \
        #         only handle networks where the n of all experiments is the same.")
        

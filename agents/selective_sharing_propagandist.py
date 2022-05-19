from agents.experimenters.binomialexperimenter import *
from typing import List

class SelectiveSharingPropagandist():
    def __init__(self):
        self.scientists: list[BinomialExperimenter] = []

    def add_scientist(self, scientist: BinomialExperimenter):
        self.scientists.append(scientist)

    # BinomialExperimenter implementation
    def get_experiment_data(self) -> list[BinomialExperiment]:
        if not self.scientists:
            return []
        experiments = [scientist.get_experiment_data() for scientist in self.scientists]
        experiments = [experiment for experiment in experiments if experiment]
        if not experiments:
            return []
        # NB this assumes that hypothesis A (which is to be promoted) is 0.5 - epsilon, and B is 0.5 + epsilon.
        experiments = [experiment for experiment in experiments if experiment.k / experiment.n < 0.5]
        if experiments:
            # Get the lowest evidence for B (i.e. highest evidence for A)
            return experiments
        else:
            return []
        # TODO: Implement safety. SelectiveSharingPropagandist cannot handle networks 
        # where n varies! (If n varies, a )
        # if not (experiments.count(experiments[0].n) == len(experiments)):
        #     raise NotImplementedError("SelectiveSharingPropagandist can currently \
        #         only handle networks where the n of all experiments is the same.")
        

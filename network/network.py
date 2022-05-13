from agents.binomialethicalscientist import BinomialEthicalScientist
import numpy as np
from enum import Enum, auto
from typing import List

class ENetworkType(Enum):
   COMPLETE = auto()
   WHEEL = auto()
   CYCLE = auto()

class ENetworkForBinomialUpdating():
    def __init__(self,
                 scientist_popcount: int,
                 scientist_network_type: ENetworkType,
                 n_per_round: int,
                 epsilon: float,
                 stop_threshold: float,
                 rng: np.random.Generator):
        self.scientist_popcount = scientist_popcount
        self.scientist_network_type = scientist_network_type
        self.scientists = [BinomialEthicalScientist(
            n_per_round,
            epsilon,
            stop_threshold,
            rng.uniform()
            ) for _ in range(scientist_popcount)]
        # Uniform function is half-open: includes low, excludes high. 
        # TODO: Wouldn't it be a good idea to exclude 0 as well? But practically, for the current
        # experiments, there is perhaps no difference – if credence is below 0.5, the scientist already
        # stops experimenting and does not learn anything new. Except – the literature here seems to 
        # assume that once stopped, always stopped. Shouldn't they keep updating on others' results even 
        # if they themselves are no longer experimenting?
        self._structure_scientific_network(self.scientists, scientist_network_type)

    ## Interface
    def enetwork_play_round(self):
        for scientist in self.scientists:
            scientist.decide_round_research_action()

    ## Private methods
    def _structure_scientific_network(self,
                                      bayes_updaters: List[BinomialEthicalScientist],
                                      network_type: ENetworkType):
        match network_type:
            case ENetworkType.COMPLETE:
                for updater in bayes_updaters:
                    self._add_all_bayes_influencers_for_updater(updater, bayes_updaters)
            case ENetworkType.WHEEL:
                print("We are wheeling it.")  # Not implemented for now
            case ENetworkType.CYCLE:
                for i, updater in enumerate(bayes_updaters):
                    self._add_cycle_bayes_influencers_for_updater(updater, i, bayes_updaters)
            case _:
                print("Invalid. All ENetworkType need to be specifically matched.")

    def _add_all_bayes_influencers_for_updater(self,
                                          updater: BinomialEthicalScientist,
                                          experimenters: List[BinomialEthicalScientist]):
        for experimenter in experimenters:
            updater.add_bayes_influencer(experimenter)
    
    def _add_cycle_bayes_influencers_for_updater(self, 
                                                updater: BinomialEthicalScientist,
                                                i: int,
                                                experimenters: List[BinomialEthicalScientist]):
        updater.add_bayes_influencer(experimenters[i-1])
        updater.add_bayes_influencer(experimenters[i])
        updater.add_bayes_influencer(experimenters[(i + 1) % self.scientist_popcount])

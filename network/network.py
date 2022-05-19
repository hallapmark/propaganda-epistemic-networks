from agents.binomialethicalscientist import BinomialEthicalScientist
from agents.bayesianupdaters.bayesianbinomialupdater import BayesianBinomialUpdater
from agents.experimenters.binomialexperimenter import BinomialExperiment
from agents.selective_sharing_propagandist import SelectiveSharingPropagandist
from sim.sim_models import *
import numpy as np
from typing import List, Optional

class ENetworkForBinomialUpdating():
    def __init__(self,
                 rng: np.random.Generator,
                 scientist_popcount: int,
                 scientist_network_type: ENetworkType,
                 n_per_round: int,
                 epsilon: float,
                 scientist_stop_threshold: float,
                 passive_updaters_config: Optional[ENPassiveUpdatersConfig],
                 selective_propagandist_active: bool):
        self.scientist_popcount = scientist_popcount
        self.scientist_network_type = scientist_network_type
        self.scientists = [BinomialEthicalScientist(
            rng,
            n_per_round,
            epsilon,
            scientist_stop_threshold,
            rng.uniform()
            # Uniform function is half-open: includes low, excludes high. 
            # TODO: Wouldn't it be a good idea to exclude 0 as well? 
            ) for _ in range(scientist_popcount)]
        self._structure_scientific_network(self.scientists, scientist_network_type)
        self.passive_updaters: Optional[List[BayesianBinomialUpdater]] = None
        if passive_updaters_config:
            self._passive_udpaters_init(passive_updaters_config, epsilon, rng)
        self.propagandist = SelectiveSharingPropagandist() if selective_propagandist_active else None
        if self.propagandist:
            self._propagandist_init(self.propagandist)

    ## Init helpers
    def _structure_scientific_network(self,
                                      bayes_updaters: List[BinomialEthicalScientist],
                                      network_type: ENetworkType):
        match network_type:
            case ENetworkType.COMPLETE:
                for updater in bayes_updaters:
                    self._add_all_bayes_influencers_for_updater(updater, bayes_updaters)
            case ENetworkType.CYCLE:
                for i, updater in enumerate(bayes_updaters):
                    self._add_cycle_bayes_influencers_for_updater(updater, i, bayes_updaters)
            case _:
                print("Invalid. All ENetworkType need to be specifically matched.")
                raise NotImplementedError

    def _passive_udpaters_init(self,
                               passive_updaters_config: ENPassiveUpdatersConfig,
                               epsilon: float,
                               rng: np.random.Generator):
        for _ in range(passive_updaters_config.updater_count):
            self._add_passive_updater(passive_updaters_config, epsilon, rng)
        if not self.passive_updaters:
            return
        for updater in self.passive_updaters:
            self._add_bayes_influencers_for_passive_updater(updater, 
                                                            passive_updaters_config, 
                                                            self.scientists)

    def _propagandist_init(self, propagandist: SelectiveSharingPropagandist):
        self._add_scientist_pool_for_propagandist(propagandist)
        if self.passive_updaters:
            self._add_propagandist_influencer_for_passive_updaters(propagandist,
                                                                   self.passive_updaters)

    ## Interface
    def enetwork_play_round(self):
        for scientist in self.scientists:
            scientist.decide_round_research_action()
        if self.passive_updaters:
            for passive_updater in self.passive_updaters:
                passive_updater.bayes_update_credence()

    def passive_updaters_avg_credence(self) -> Optional[float]:
        if not self.passive_updaters:
            return None
        return np.mean([a.credence for a in self.passive_updaters])
        
    ## Private methods
    # TODO: The influencer logic can probably be made more generic
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
    
    def _add_bayes_influencers_for_passive_updater(self,
                                                   updater: BayesianBinomialUpdater,
                                                   passive_updater_config: ENPassiveUpdatersConfig,
                                                   experimenters: List[BinomialEthicalScientist]):
        for i in range(passive_updater_config.scientist_influencer_count):
            if len(experimenters) > i:
                updater.add_bayes_influencer(experimenters[i])

    def _add_passive_updater(self,
                             passive_updaters_config: ENPassiveUpdatersConfig,
                             epsilon: float,
                             rng: np.random.Generator):
        min_p = passive_updaters_config.min_prior
        max_p = passive_updaters_config.max_prior
        updater = BayesianBinomialUpdater(epsilon=epsilon,
                                          prior=rng.uniform(min_p, max_p))
        if self.passive_updaters:
            self.passive_updaters.append(updater)
        else:
            self.passive_updaters = [updater]

    def _add_scientist_pool_for_propagandist(self, propagandist: SelectiveSharingPropagandist):
        for scientist in self.scientists:
            propagandist.add_scientist(scientist)
        
    def _add_propagandist_influencer_for_passive_updaters(self, 
                                                        propagandist: SelectiveSharingPropagandist, 
                                                        passive_updaters: List[BayesianBinomialUpdater]):
        for updater in passive_updaters:
            updater.add_bayes_influencer(propagandist)


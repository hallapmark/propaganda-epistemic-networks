from network.network import ENetworkForBinomialUpdating
import numpy as np
from typing import Optional
from sim.sim_models import ENSimulationRawResults

class EpistemicNetworkSimulation():
    def __init__(self,
                 epistemic_network: ENetworkForBinomialUpdating,
                 maxrounds: int,
                 low_stop: float,
                 high_stop: float):
        self.epistemic_network = epistemic_network
        self._low_stop = low_stop
        self._maxrounds = maxrounds
        self._high_stop = high_stop
        self._sim_round = 0
        self.results: Optional[ENSimulationRawResults] = None
    
    def run_sim(self):
        for i in range(1, self._maxrounds + 1):
            if self.results:
                break
            self._sim_action(i)
        if not self.results:
            self.results = ENSimulationRawResults(None, None, self._sim_round, None)

    def _sim_action(self, sim_round: int):
        if self.results:
            return
        self._sim_round = sim_round
        credences = np.array([a.credence for a in self.epistemic_network.scientists])
        if all(credences < self._low_stop):
            # Everyone's credence in B is below 0.5. Abandon further research
            self.results = ENSimulationRawResults(None, sim_round, sim_round, None)
            return
        if all(credences > .99):
            # Everyone's credence in B is above .99. Scientific consensus reached
            p_avg_cr = self.epistemic_network.passive_updaters_avg_credence()
            self.results = ENSimulationRawResults(sim_round,
                                                  None,
                                                  sim_round,
                                                  p_avg_cr)
            # get average credence that policymakers have when scientists reach consensus 

            return
        self.epistemic_network.enetwork_play_round()

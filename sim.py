from network.network import ENetworkForBinomialUpdating
import numpy as np
from typing import Optional

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
        self.consensus_round: Optional[int] = None
        self.abandon_round: Optional[int] = None
    
    def run_sim(self):
        for i in range(self._maxrounds):
            if self.abandon_round or self.consensus_round:
                break
            self.sim_action(i)

    def sim_action(self, sim_round: int):
        self._sim_round = sim_round
        if self.consensus_round or self.abandon_round:
            return
        credences = np.array([a.credence for a in self.epistemic_network.scientists])
        if all(credences < self._low_stop):
            # Everyone's credence in B is below 0.5. Abandon further research
            self.abandon_round = sim_round + 1
            return
        if all(credences > .99):
            # Everyone's credence in B is above .99. Scientific consensus reached
            self.consensus_round = sim_round + 1
            return
        self.epistemic_network.enetwork_play_round()



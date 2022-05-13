import numpy as np
from network.network import EpistemicNetworkForBinomialUpdating, ENetworkType

if __name__ == "__main__":
    rng = np.random.default_rng(253591)
    # params = [10, ENetworkType.COMPLETE, 100, 0.01, 0.5, rng]
    network = EpistemicNetworkForBinomialUpdating(10, ENetworkType.COMPLETE, 100, 0.01, 0.5, rng)
    test_scientist = network.scientists[0]
    test_scientist2 = network.scientists[1]
    print(f'test_scientist. Value of self.binomial_experiment before the experiment: {test_scientist.binomial_experiment}')
    print(f'test_scientist2. Value of self.binomial_experiment before the experiment: {test_scientist2.binomial_experiment}')
    network.enetwork_play_round()
    print(f'test_scientist. Value of self.binomial_experiment after the experiment: {test_scientist.binomial_experiment}')
    print(f'test_scientist2. Value of self.binomial_experiment after the experiment: {test_scientist2.binomial_experiment}')
    network.enetwork_play_round()
    print(f'test_scientist. Value of self.binomial_experiment after another experiment: {test_scientist.binomial_experiment}')
    print(f'test_scientist2. Value of self.binomial_experiment after another experiment: {test_scientist2.binomial_experiment}')
    network.enetwork_play_round()
    print(f'test_scientist. Value of self.binomial_experiment after another experiment: {test_scientist.binomial_experiment}')
    print(f'test_scientist2. Value of self.binomial_experiment after another experiment: {test_scientist2.binomial_experiment}')

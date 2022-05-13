import numpy as np
import timeit
from network.network import ENetworkForBinomialUpdating, ENetworkType
from sim import EpistemicNetworkSimulation

if __name__ == "__main__":
    rng = np.random.default_rng(253591)
    # params = [10, ENetworkType.COMPLETE, 100, 0.01, 0.5, rng]
    abandon_count = 0
    consensus_count = 0
    start_time = timeit.default_timer()
    for _ in range(1000):
        network = ENetworkForBinomialUpdating(10, ENetworkType.CYCLE, 1000, 0.001, 0.5, rng)
        simulation = EpistemicNetworkSimulation(network, 10000, 0.5, 0.99)
        simulation.run_sim()
        if simulation.abandon_round:
            abandon_count += 1
        elif simulation.consensus_round:
            consensus_count += 1
    stop_time = timeit.default_timer()
    print(f'Abandon count: {abandon_count}')
    print(f'Consensus count: {consensus_count}')
    print(f'Sims start: {start_time}')
    print(f'Sims stop: {stop_time}')
    
    # test_scientist = network.scientists[0]
    # test_scientist2 = network.scientists[1]
    # print(f'test_scientist. Value of self.binomial_experiment before the experiment: {test_scientist.binomial_experiment}')
    # print(f'test_scientist2. Value of self.binomial_experiment before the experiment: {test_scientist2.binomial_experiment}')
    # network.enetwork_play_round()
    # print(f'test_scientist. Value of self.binomial_experiment after the experiment: {test_scientist.binomial_experiment}')
    # print(f'test_scientist2. Value of self.binomial_experiment after the experiment: {test_scientist2.binomial_experiment}')
    # network.enetwork_play_round()
    # print(f'test_scientist. Value of self.binomial_experiment after another experiment: {test_scientist.binomial_experiment}')
    # print(f'test_scientist2. Value of self.binomial_experiment after another experiment: {test_scientist2.binomial_experiment}')
    # network.enetwork_play_round()
    # print(f'test_scientist. Value of self.binomial_experiment after another experiment: {test_scientist.binomial_experiment}')
    # print(f'test_scientist2. Value of self.binomial_experiment after another experiment: {test_scientist2.binomial_experiment}')

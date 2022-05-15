import numpy as np
import timeit
from multiprocessing import Pool
from network.network import ENetworkForBinomialUpdating, ENetworkType
from sim import *
from typing import Optional, List

def run_sim(scientist_pop_count: int, 
            network_type: ENetworkType, 
            n_per_round: int, 
            epsilon: float, 
            scientist_stop_threshold: float,
            rng: np.random.Generator,
            max_research_rounds: int,
            consensus_threshold: float) -> Optional[ENSimulationResults]:
    network = ENetworkForBinomialUpdating(scientist_pop_count,
                                          network_type,
                                          n_per_round,
                                          epsilon,
                                          scientist_stop_threshold,
                                          rng)
    simulation = EpistemicNetworkSimulation(network, 
                                            max_research_rounds, 
                                            scientist_stop_threshold, 
                                            consensus_threshold)
    simulation.run_sim()
    return simulation.results

# A partial reproduction of the results of https://philpapers.org/rec/ZOLTCS
# We'll use this to verify our model before moving on to more complex networks.
def zollman_2007(rng_streams: List[np.random.Generator]):
    pool = Pool()
    results_from_sims = pool.starmap(
        run_sim, 
        [(10, ENetworkType.COMPLETE, 1000, 0.001, 0.5, rng, 10000, 0.99) for rng in rng_streams]
        )
    pool.close()
    # print("Hi, I am a print statement in-between pool.close() and pool.join()")
    pool.join()
    # print("Hi, I am a print statement after pool.join()")
    # print()
    # print("Results from starmap!:")
    # # print(results_from_sims)
    # print()
    # forloopsimresults = []
    # for rng in streams:
    #     network = ENetworkForBinomialUpdating(10, ENetworkType.COMPLETE, 1000, 0.001, 0.5, rng)
    #     simulation = EpistemicNetworkSimulation(network, 10000, 0.5, 0.99)
    #     simulation.run_sim()
    #     forloopsimresults.append(simulation.results)
    # time_elapsed = timeit.default_timer() - start_time
    # print("For loop sim results:")
    # print(forloopsimresults)
    # print(f'Time elapsed: {time_elapsed}s')

def main():
    # freeze_support() # Only needed if we're making an executable that packs Python with it
    # https://docs.python-guide.org/shipping/freezing/
    sim_count = 1800
    #params = [(10, ENetworkType.COMPLETE, 1000, 0.001, 0.5, rng, 10000, 0.99) for _ in range(sim_count)]
    
    # Careful when passing rng instances to starmap. If you do not set independent seeds, you will get 
    # the *same* results each simulation since the subprocesses share the parent's initial rng state.
    # https://numpy.org/doc/stable/reference/random/parallel.html
    child_seeds = np.random.SeedSequence(25359).spawn(sim_count)
    rng_streams = [np.random.default_rng(s) for s in child_seeds]
    start_time = timeit.default_timer()
    zollman_2007(rng_streams)
    time_elapsed = timeit.default_timer() - start_time

if __name__ == "__main__":
    main()
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

import numpy as np
import timeit
from multiprocessing import Pool
from network.network import ENetworkForBinomialUpdating, ENetworkType
from sim import *
from typing import Dict, Optional, List
import os
import csv

class ENParams(NamedTuple):
    scientist_pop_count: int 
    network_type: ENetworkType
    binom_n_per_round: int
    epsilon: float
    scientist_stop_threshold: float
    max_research_rounds_allowed: int
    consensus_threshold: float

class ENSimsReadableSummary(NamedTuple):
    proportion_consensus_reached: str
    avg_consensus_round: str

class ENResultsSummary(NamedTuple):
    params: ENParams
    sims_summary: ENSimsReadableSummary


def run_sim(scientist_pop_count: int, 
            network_type: ENetworkType, 
            n_per_round: int, 
            epsilon: float, 
            scientist_stop_threshold: float,
            max_research_rounds: int,
            consensus_threshold: float,
            rng: np.random.Generator) -> Optional[ENSimulationResults]:
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
def zollman_2007(params: ENParams, rng_streams: List[np.random.Generator]) -> ENResultsSummary:
    if not rng_streams:
        raise ValueError("There needs to be at least one rng.")
    pool = Pool()
    results_from_sims = pool.starmap(run_sim,
                                     [params + (rng,) for rng in rng_streams])
    pool.close()
    pool.join()
    if None in results_from_sims:
        raise ValueError("Failed to get results from at least one simulation.")
    results = [r for r in results_from_sims if r is not None]
    cons_count = consensus_count(results)
    proportion_consensus_reached = round(cons_count / len(results), 3)
    if cons_count > 0:
        # TODO: Add try statement?
        avg_c_r = np.mean([res.consensus_round for res in results if res.consensus_round])
        avg_consensus_round = round(float(avg_c_r), 3)
    else:
        avg_consensus_round = "N/A"
    sims_summary = ENSimsReadableSummary(str(proportion_consensus_reached), str(avg_consensus_round))
    return ENResultsSummary(params, sims_summary)

def consensus_count(results: List[ENSimulationResults]) -> int:
    total = 0
    for sim_result in results:
        if not sim_result:
            continue
        if sim_result.consensus_round:
            total += 1
    return total

# def zollman_2007_proportion_of_success():
#     for 

# ['graph', 'agents', 'epochs', 'conclusion', 'trials', 'epsilon', 'mistrust']
def record_sim(params: List[str], sim_time: str, results: List[str], path: str, headers: List[str]):
    file_exists = os.path.isfile(path)
    with open(path, newline='', mode = 'a') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(headers)
        output_list = params
        for sim_result in results:
            output_list.append(sim_time)
            output_list.append(sim_result)
        writer.writerows([output_list])

def main():
    sim_count = 100 # TODO: This is standardly 10000
    #params = [(10, ENetworkType.COMPLETE, 1000, 0.001, 0.5, rng, 10000, 0.99) for _ in range(sim_count)]
    
    # Careful when passing rng instances to starmap. If you do not set independent seeds, you will get 
    # the *same* results each simulation since the subprocesses share the parent's initial rng state.
    # https://numpy.org/doc/stable/reference/random/parallel.html
    child_seeds = np.random.SeedSequence(25359).spawn(sim_count)
    rng_streams = [np.random.default_rng(s) for s in child_seeds]
    start_time = timeit.default_timer()
    zollman2007_configs = [(pop, ENetworkType.COMPLETE, 1000, 0.001, 0.5, 10000, 0.99) for pop in range(3, 4)]
    for param_config in zollman2007_configs:
        params = ENParams(*param_config)
        zollman_results = zollman_2007(params, rng_streams)
        time_elapsed = timeit.default_timer() - start_time
        print(f'Time elapsed: {time_elapsed}s')
        headers = [param_name for param_name in params._asdict().keys()]
        headers.append(f'sim time (s)')
        for summary_field in zollman_results.sims_summary._asdict().keys():
            headers.append(summary_field)
        print(headers)
        param_str_list = [str(param) for param in param_config]
        result_str_list = [zollman_results.sims_summary.proportion_consensus_reached,
                    zollman_results.sims_summary.avg_consensus_round]
        #path = os.path.join(os.getcwd(), 'zollman2007.csv')
        print(param_str_list)
        print(result_str_list)
        record_sim(param_str_list, str(time_elapsed), result_str_list, "zollman2007.csv", headers)

if __name__ == "__main__":
    # freeze_support() # Only needed if we're making an executable that packs Python with it
    # https://docs.python-guide.org/shipping/freezing/
    main()
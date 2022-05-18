import numpy as np
import timeit
from multiprocessing import Pool
from network.network import ENetworkForBinomialUpdating, ENetworkType
from sim.sim import *
from sim.sim_models import *
from agents.bayesianupdaters.bayesianbinomialupdater import BayesianBinomialUpdater
from typing import Optional, List
from enum import Enum, auto
import os
import csv

class ENSimType(Enum):
    ZOLLMAN_COMPLETE = auto()
    ZOLLMAN_CYCLE = auto()
    POLICYMAKERS_COMPLETE = auto()
    POLICYMAKERS_CYCLE = auto()
    PROPAGANDA_COMPLETE = auto()
    PROPAGANDA_CYCLE = auto()
    COUNTER_PROPAGANDA_COMPLETE = auto()
    COUNTER_PROPAGANDA_CYCLE = auto()

class ENSimSetup():
    def __init__(self,
                 sim_count: int,
                 sim_type: Optional[ENSimType]):
        self.sim_count = sim_count
        self.sim_type = sim_type
    
    def quick_setup(self):
        """ Setup sims from pre-defined templates, e.g. ENSimType.ZOLLMAN_COMPLETE. 
        Use setup_sims instead if you need to customize the parameters."""
        if not self.sim_type:
            raise ValueError("Quick setup can only be called if you have specified ENSimType")
        match self.sim_type:
            # A partial reproduction of the results of Zollman https://philpapers.org/rec/ZOLTCS
            case ENSimType.ZOLLMAN_COMPLETE:
                configs = [ENParams(pop, ENetworkType.COMPLETE, 1000, 0.001, 0.5, 10000, 0.99, None, False) for pop in range(3, 5)]
                self.setup_sims(configs, "zollman2007.csv")
            case ENSimType.ZOLLMAN_CYCLE:
                configs = [ENParams(pop, ENetworkType.CYCLE, 1000, 0.001, 0.5, 10000, 0.99, None, False) for pop in range(4, 5)]
                self.setup_sims(configs, "zollman2007.csv")
            case ENSimType.POLICYMAKERS_COMPLETE: # Weatherall et al. 2020 (without propagandists)   pop = 4; 6; 10; 20; 50 – but we are skipping 50 for sure
                configs = [ENParams(pop, ENetworkType.COMPLETE, 1000, 0.001, 0.5, 10000, 0.99, ENPassiveUpdatersConfig(2, 0, 0.5, infl_count), False) for pop in (4, 6)
                                                                                                                                               for infl_count in range(1, pop+1)]
                self.setup_sims(configs, "policymakers_complete.csv")
            case ENSimType.POLICYMAKERS_CYCLE: # Figure 2 first part
                configs = [ENParams(pop, ENetworkType.CYCLE, 10, 0.05, 0.5, 10000, 0.99, ENPassiveUpdatersConfig(2, 0, 0.5, infl_count), False)    for pop in (20,) 
                                                                                                                                            for infl_count in range(1, pop+1)]
                self.setup_sims(configs, "policymakers_cycle.csv")                                          
            case ENSimType.PROPAGANDA_COMPLETE:
                raise NotImplementedError
            case ENSimType.PROPAGANDA_CYCLE:
                configs = [ENParams(pop, ENetworkType.CYCLE, 10, 0.05, 0.5, 10000, 0.99, ENPassiveUpdatersConfig(2, 0, 0.5, infl_count), True)  for pop in (20,) 
                                                                                                                                                for infl_count in range(1, pop+1)]
                self.setup_sims(configs, "policymakers_cycle.csv") 
            case ENSimType.COUNTER_PROPAGANDA_COMPLETE:
                raise NotImplementedError
            case ENSimType.COUNTER_PROPAGANDA_CYCLE:
                raise NotImplementedError

    def setup_sims(self, configs: List[ENParams], output_filename: str):
        # We need to be careful when passing rng instances to starmap. If we do not set independent seeds, 
        # we will get the *same* results each simulation since the subprocesses share the parent's initial 
        # rng state.
        # https://numpy.org/doc/stable/reference/random/parallel.html
        child_seeds = np.random.SeedSequence(25359).spawn(self.sim_count)
        rng_streams = [np.random.default_rng(s) for s in child_seeds]
        for param_config in configs:
            print(f'Running config: {param_config}')
            print('...')
            start_time = timeit.default_timer()
            results_summary = self.run_sims_for_param_config(param_config, rng_streams)
            time_elapsed = timeit.default_timer() - start_time
            print(f'Time elapsed: {time_elapsed}s')
            csv_data = self.data_for_writing(results_summary, self.sim_count, time_elapsed)
            print()
            self.record_sim(csv_data, output_filename)

    def run_sims_for_param_config(self, params: ENParams, rng_streams: List[np.random.Generator]) -> ENSimsSummary:
        if not rng_streams:
            raise ValueError("There needs to be at least one rng.")
        pool = Pool()
        results_from_sims = pool.starmap(self.run_sim,
                                        [(rng,) + params for rng in rng_streams])
        pool.close()
        pool.join()
        if None in results_from_sims:
            raise ValueError("Failed to get results from at least one simulation.")
        results = [r for r in results_from_sims if r is not None]
        cons_count = self.consensus_count(results)
        proportion_consensus_reached = round(cons_count / len(results), 3)
        if cons_count > 0:
            # TODO: Add try statement?
            avg_c_r = np.mean([res.consensus_round for res in results if res.consensus_round])
            avg_consensus_round = round(float(avg_c_r), 3)
        else:
            avg_consensus_round = "N/A"
        passive_upd_averages = [res.passive_updaters_avg_credence for res in results if res.passive_updaters_avg_credence]
        passive_cr = "N/A"
        if passive_upd_averages:
            passive_cr = np.mean(passive_upd_averages)
        sims_summary = ENResultsSummary(str(proportion_consensus_reached), str(avg_consensus_round), str(passive_cr))
        return ENSimsSummary(params, sims_summary)

    def run_sim(self,
                rng: np.random.Generator,
                scientist_pop_count: int, 
                network_type: ENetworkType, 
                n_per_round: int, 
                epsilon: float, 
                scientist_stop_threshold: float,
                max_research_rounds: int,
                consensus_threshold: float,
                passive_updaters_config: Optional[ENPassiveUpdatersConfig],
                propagandist_active: bool) -> Optional[ENSimulationRawResults]:
        network = ENetworkForBinomialUpdating(rng,
                                              scientist_pop_count,
                                              network_type,
                                              n_per_round,
                                              epsilon,
                                              scientist_stop_threshold,
                                              passive_updaters_config,
                                              propagandist_active)
        simulation = EpistemicNetworkSimulation(network, 
                                                max_research_rounds, 
                                                scientist_stop_threshold, 
                                                consensus_threshold)
        simulation.run_sim()
        return simulation.results

    def consensus_count(self, results: List[ENSimulationRawResults]) -> int:
        total = 0
        for sim_result in results:
            if not sim_result:
                continue
            if sim_result.consensus_round:
                total += 1
        return total

    def record_sim(self, results: ENResultsCSVWritableSummary, path: str):
        file_exists = os.path.isfile(path)
        # res_dir = "/results"
        # Path(res_dir).mkdir(parents=True, exist_ok=True)
        # filename = Path(res_dir, filename).with_suffix('.csv')
        with open(path, newline='', mode = 'a') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(results.headers)
            writer.writerow(results.sim_data)

    def data_for_writing(self, 
                        sims_summary: ENSimsSummary, 
                        sim_count: int, 
                        time_elapsed: float) -> ENResultsCSVWritableSummary:
        headers = ['Sim count']
        headers.extend([param_name for param_name in sims_summary.params._asdict().keys()])
        headers.append('sim time (s)')
        summary_fields = [field for field in sims_summary.results_summary._asdict().keys()]
        headers.extend(summary_fields)

        sim_data = [str(sim_count)]
        sim_data.extend([str(param_val) for param_val in sims_summary.params])
        sim_data.append(str(time_elapsed))
        result_str_list = [r for r in sims_summary.results_summary]
        print(f'Summary fields: {summary_fields}')
        print(f'Results from config: {result_str_list}')
        sim_data.extend(result_str_list)
        summary = ENResultsCSVWritableSummary(headers, sim_data)
        return summary

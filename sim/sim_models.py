from typing import Optional, NamedTuple, List
from network.network import ENetworkType

class ENParams(NamedTuple):
    scientist_pop_count: int 
    network_type: ENetworkType
    binom_n_per_round: int
    epsilon: float
    scientist_stop_threshold: float
    max_research_rounds_allowed: int
    consensus_threshold: float
    passive_updaters_count: int # E.g. policymakers

class ENSimulationRawResults(NamedTuple):
    consensus_round: Optional[int]
    research_abandoned_round: Optional[int]
    final_sim_round: int

class ENResultsSummary(NamedTuple):
    proportion_consensus_reached: str
    avg_consensus_round: str

class ENSimsSummary(NamedTuple):
    params: ENParams
    results_summary: ENResultsSummary

class ENResultsCSVWritableSummary(NamedTuple):
    headers: List[str]
    sim_data: List[str]
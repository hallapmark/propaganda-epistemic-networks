from typing import Optional, NamedTuple,  List
from enum import Enum, auto

class ENetworkType(Enum):
   COMPLETE = auto()
   CYCLE = auto()
class ENPassiveUpdatersConfig(NamedTuple):
    updater_count: int
    min_prior: float
    max_prior: float
    scientist_influencer_count: int

class ENParams(NamedTuple):
    scientist_pop_count: int 
    network_type: ENetworkType
    binom_n_per_round: int
    epsilon: float
    scientist_stop_threshold: float
    max_research_rounds_allowed: int
    consensus_threshold: float
    passive_updaters_config: Optional[ENPassiveUpdatersConfig] # E.g. policymakers

class ENSimulationRawResults(NamedTuple):
    consensus_round: Optional[int]
    research_abandoned_round: Optional[int]
    final_sim_round: int
    passive_updaters_avg_credence: Optional[float]

class ENResultsSummary(NamedTuple):
    scientist_proportion_consensus_reached: str
    scientists_avg_consensus_round: str
    passive_updaters_avg_credence: str

class ENSimsSummary(NamedTuple):
    params: ENParams
    results_summary: ENResultsSummary

class ENResultsCSVWritableSummary(NamedTuple):
    headers: List[str]
    sim_data: List[str]
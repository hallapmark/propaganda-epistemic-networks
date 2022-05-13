from typing import Protocol, NamedTuple, Optional

class BinomialExperiment(NamedTuple): 
    k: int
    n: int

class BinomialExperimenter(Protocol):
    def get_experiment_data() -> Optional[BinomialExperiment]: # type: ignore (silence Pylance)
        """ Get the data from the latest experiment, k: int and n: int."""
    

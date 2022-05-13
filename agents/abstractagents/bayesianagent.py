from agents.abstractagents.doxasticagent import DoxasticAgent
from abc import abstractmethod

class BayesianAgent(DoxasticAgent):
    def __init__(self, **kw):
        print(f"Init BayesianAgent.")
        super().__init__(**kw)
    
    @abstractmethod
    def bayes_update_credence(self):
        """ Updates the agent's credence in H given certain evidence E using Bayes' rule:
        posterior = (credence * likelihood) / marginal_likelihood."""
        return NotImplementedError

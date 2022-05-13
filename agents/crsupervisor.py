from agents.abstractagents.doxasticagent import DoxasticAgent
from abc import abstractmethod

class CredenceBasedSupervisor(DoxasticAgent):
    def __init__(self, stop_threshold: float, **kw):
        super().__init__(**kw)
        self._stop_threshold = stop_threshold

    def decide_round_research_action(self):
        if self.credence < self._stop_threshold:
            # TODO: Remove print statements
            print(f"Credence {self.credence} in hypothesis is low. Stop experimenting!")
            self._stop_action()
        else:
            print(f"Credence {self.credence} in hypothesis is over the threshold. Continue research!")
            self._continue_action()
    
    @abstractmethod
    def _stop_action(self):
        raise NotImplementedError 
    
    @abstractmethod
    def _continue_action(self):
        raise NotImplementedError
    
    @abstractmethod
    def _finally(self):
        raise NotImplementedError

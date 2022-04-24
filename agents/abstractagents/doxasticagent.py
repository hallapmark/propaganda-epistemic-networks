from abc import ABC

class DoxasticAgent(ABC):
    def __init__(self, prior: float):
        print(f"Init DoxasticAgent. I should have a credence! {prior}")
        self.credence: float = prior
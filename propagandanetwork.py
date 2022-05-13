from agents.binomialethicalscientist import BinomialEthicalScientist
import numpy as np

if __name__ == "__main__":
    def createScientist(rng: np.random.Generator) -> BinomialEthicalScientist:
        return BinomialEthicalScientist(n_per_round=100,
                                        epsilon=0.01,
                                        stop_threshold=0.5,
                                        prior=rng.uniform())
    scientist_popsize = 10
    rng = np.random.default_rng(253591)
    priors = []
    # Uniform function is half-open: includes low, excludes high. 
    # TODO: Wouldn't it be a good idea to exclude 0 as well? But practically, for the current
    # experiments, there is perhaps no difference – if credence is below 0.5, the scientist already
    # stops experimenting and does not learn anything new. Except – the literature here seems to 
    # assume that once stopped, always stopped. Shouldn't they keep updating on others' results even 
    # if they themselves are no longer experimenting?
    #scientists = [createScientist(rng) for _ in range(scientist_popsize)]
    #print(scientists[2].credence)
    scientist = BinomialEthicalScientist(n_per_round=100,
                                        epsilon=0.01,
                                        stop_threshold=0.5,
                                        prior=rng.uniform())


from agents.abstractagents.bayesianagent import BayesianAgent
from typing import Optional

""" A Bayesian updater who knows how to update on binomial distributions."""
# TODO: Following the Zollman (2007) literature, this currently assumes that there 
# are only two competing hypotheses/possible worlds: H := p = 1 + epsilon and 
# not-H := p = 1-epsilon where p is the probability that a binary event happens. 
# In the future, should probably generalize this to also work with a full distribution 
# of hypotheses (see also beta distribution.)
class BayesianBinomialUpdater(BayesianAgent):
    def __init__(self, k: Optional[int], n: int, epsilon: float, **kw):
        print(f"Init BayesianBinomialUpdater.")
        super().__init__(**kw)
        # k is optional because some binomial updaters will run their own experiments after 
        # initialization
        self.k: Optional[int] = k 
        self.n = n
        self.epsilon = epsilon

    # Public interface
    # Superclass mandatory method implementation
    def bayes_update_credence(self) -> bool:
        prior = self.credence
        k = self.k
        n = self.n 
        p = 0.5 + self.epsilon
        if k:
            self.credence = self._bayes_calculate_posterior_two_possible_worlds(prior, k, n, p)
            return True
        return False
    
    # Private methods
    def _bayes_calculate_posterior_two_possible_worlds(self, prior: float, k: int, n: int, p: float) -> float:
        """ Truncated Bayes' formula for the binomial distribution. It is assumed that there 
        are only two possible parameter values (two possible worlds): p and 1-p. This parameter gives
        the probability of a "success" event occurring on a given try. """
        # This formula looks bonkers but it is derived from the full Bayes' formula (as applied to binomials).
        # This speeds up the simulations.
        return 1 / (1 + ((1 - prior) * ((1 - p) / p) ** (2 * k - n)) / prior)
 
    # P(E|H)
    # def _calculate_likelihood(self, k, n, p, factorial_func) -> float:
    #     """ Use the full likelihood formula if you are either ignoring the denominator 
    #     in Bayes' theorem or if you need to calculate P(E|H) independently for some 
    #     reason. Otherwise, use the truncated likelihood function or just."""
    #     f = factorial_func
    #     return f(n) / (f(k) * f(n-k)) * p ** k * (1 - p) ** (n - k)
    
    # # P(E|H) when some terms cancel out in the denominator and numerator of Bayes' theorem
    # def _calculate_truncated_likelihood(self, k: int, n: int, p: float) -> float:
    #     """ Use this only if you are calculating the posterior, i.e. you are 
    #     not ignoring the denominator. You can use the fact that some terms cancel 
    #     out from the denominator and numerator to simplify the likelihood formula."""
    #     # Note: This simplifies even further in Bayes' theorem. But we can use this function
    #     # in Jeffrey updating.
    #     return p ** k * (1 - p) ** (n - k)

    # def calculate_truncated_p_E_nH_two_possible_worlds(self, k: int, n: int, p: float) -> float:
    #     """ Calculate P(E|~H) for the binomial distribution when there are only two possible 
    #     parameter values/two possible worlds: p and 1-p."""
    #     return (1-p) ** k * p ** (n - k)

    # # P(E)
    # def _calculate_marginal_likelihood_two_possible_worlds(self,
    #                                                        prior: float,
    #                                                        p_E_H: float,
    #                                                        p_E_nH: float) -> float:
    #     return prior * p_E_H + (1 - prior) * p_E_nH
    
    # TODO: Make this into a proper unit test
    def verify_update(self):
        k = self.k
        n = self.n
        if k:
            # 1 / (1 + (1 - self.credence) * ((1-p / p) ** (2 * k - n)) / self.credence)
            # From https://github.com/jweisber/sep-sen.  
            self.credence = 1 / (1 + (1 - self.credence) * (((0.5 - self.epsilon) / (0.5 + self.epsilon)) ** (2 * k - n)) / self.credence)

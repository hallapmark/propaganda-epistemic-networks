from agents.abstractagents.bayesianagent import BayesianAgent
from typing import Optional
from math import factorial

""" A Bayesian updater who knows how to update on binomial distributions."""
class BayesianBinomialUpdater(BayesianAgent):
    def __init__(self, k: Optional[int], n: int, epsilon: float, **kw):
        print(f"Init BayesianBinomialUpdater.")
        super().__init__(**kw)
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
            self.credence = self._bayes_calculate_posterior(prior, k, n, p)
            return True
        return False
    
    # Private methods
    def _bayes_calculate_posterior(self, prior: float, k: int, n: int, p: float) -> float:
        likelihood = self._calculate_truncated_likelihood(k, n, p)
        p_e_nH = self._p_E_nH_truncated(k, n, p)
        marginal_likelihood = self._calculate_marginal_likelihood(prior, likelihood, p_e_nH)
        return prior * likelihood / marginal_likelihood
 
    # P(E|H)
    def _calculate_likelihood(self, k, n, p, factorial_func) -> float:
        """ Use the full likelihood formula if you are either ignoring the denominator 
        in Bayes' theorem or if you need to calculate P(E|H) independently for some 
        reason. Otherwise, use the truncated likelihood function."""
        f = factorial_func
        return f(n) / (f(k) * f(n-k)) * p ** k * (1 - p) ** (n - k)
    
    # P(E|H) when some terms cancel out in the denominator and numerator of Bayes' theorem
    def _calculate_truncated_likelihood(self, k: int, n: int, p: float) -> float:
        """ Use this if you are calculating the full Bayesian posterior, i.e. you are 
        not ignoring the denominator. You can use the fact that some terms cancel 
        out from the denominator and numerator to simplify the likelihood formula."""
        return p ** k * (1 - p) ** (n - k)

    def _p_E_nH_truncated(self, k: int, n: int, p: float) -> float:
        return (1-p) ** k * p ** (n - k)

    # P(E) # TODO: Is this truncated as well?
    def _calculate_marginal_likelihood(self,
                                       prior: float,
                                       p_E_H: float,
                                       p_E_nH: float) -> float:
        return prior * p_E_H + (1 - self.credence) * p_E_nH
    
    # TODO: Make this into a proper unit test
    def verify_update(self):
        k = self.k
        n = self.n
        if k:
            # 1 / (1 + (1 - self.credence) * ((1-p / p) ** (2 * k - n)) / self.credence)
            # From https://github.com/jweisber/sep-sen
            self.credence = 1 / (1 + (1 - self.credence) * (((0.5 - self.epsilon) / (0.5 + self.epsilon)) ** (2 * k - n)) / self.credence)
            
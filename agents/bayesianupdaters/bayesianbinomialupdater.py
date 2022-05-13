from agents.abstractagents.bayesianagent import BayesianAgent
from experimenters.binomialexperimenter import BinomialExperimenter
from typing import List

# TODO: Following Zollman (2007) and Weatherall, O'Connor and Bruner (2020), this
# implementation currently assumes that there are only two competing hypotheses/possible worlds:
# H := p = 1 + epsilon and not-H := p = 1-epsilon where p is the probability that
# a binary event happens. In the future, should probably also include a version that
# works with a full distribution of hypotheses (see also beta distribution, and Zollman 2010.)
""" A Bayesian updater who knows how to update on binomial distributions."""
class BayesianBinomialUpdater(BayesianAgent):
    def __init__(self, epsilon: float, **kw):
        print(f"Init BayesianBinomialUpdater.")
        super().__init__(**kw)
        # k is optional because some binomial updaters will run their own experiments after
        # initialization
        self.epsilon = epsilon
        # Influencers can include self (usually, it would be odd if it did not)
        self.bayes_influencers: List[BinomialExperimenter] = []
        # In the future, we can also add jeffrey influencers whose data the updater
        # does not fully trust.
        
    def add_bayes_influencers(self, influencers: List[BinomialExperimenter]):
        for influencer in influencers:
            self.bayes_influencers.append(influencer)

    # Public interface
    # Superclass mandatory method implementation
    def bayes_update_credence(self):
        for influencer in self.bayes_influencers:
            self._bayes_update_credence_on_influencer(influencer)
        
    # Private methods
    def _bayes_update_credence_on_influencer(self, influencer: BinomialExperimenter): 
        data = influencer.get_experiment_data()
        if data:
            p = 0.5 + self.epsilon
            self.credence = self._bayes_calculate_posterior_two_possible_worlds(self.credence, data.k, data.n, p)
            
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

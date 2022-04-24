from agents.binomialethicalscientist import BinomialEthicalScientist
from agents.bayesianupdaters.bayesianbinomialupdater import BayesianBinomialUpdater

if __name__ == "__main__":
    binomial_ethical_scientist = BinomialEthicalScientist(n=100,
                                                          epsilon=0.01,
                                                          stop_threshold=0.5,
                                                          prior=0.54)
    binomial_ethical_scientist.decide_research_action()
    print(f'posterior: {binomial_ethical_scientist.credence}')
    # for x in range(200):
    #     print(f"Round {x}")
    #     binomial_ethical_scientist.decide_research_action()

    k = 56
    n = 100
    epsilon = 0.01
    prior = 0.51
    myUpdater = BayesianBinomialUpdater(k, n, epsilon, prior = prior)
    verifyingUpdater = BayesianBinomialUpdater(k, n, epsilon, prior = prior)
    testUpdater = BayesianBinomialUpdater(k, n, epsilon, prior = prior)
    if myUpdater.bayes_update_credence() == True:
        credence = myUpdater.credence
        print(f"Credence per my calculations: {credence}")
        verifyingUpdater.verify_update()
        print(f"Credence per verifying calculations: {verifyingUpdater.credence}")


# Implements a rock-paper-scissor agent that chooses its strategy 
# based on the counterfactual regret minimization algorithm:
# 1 - For each player, initialize all cumulative regrets to 0
# 2 - For some number of iterations:
#       - Compute a regret matching strategy profile (if all regret for a player
#       are non positive, use a uniform random strategy
#       - Add the strategy profile to the strategy profile sum
#       - Select each player action profile according to the strategy profile
#       - Compute player regrets
#       - Add player regrets to player cumulative regrets
#       - Return the average strategy profile, i.e. the strategy sum
#       divided by the number of iterations

import numpy as np
from matplotlib import pyplot as plt
import os
import math


class CFRMinimizer:

    log_filename = 'log.txt'

    def __init__(self, actions=('R', 'P', 'S', ), decay=1, debug=False, log=False):
        self.num_actions = len(actions)
        self.actions = actions
        self.regret_hist =  np.zeros(shape=(1, 3), dtype=float)
        self.strategy_hist = np.zeros(shape=(1, 3), dtype=float)
        self.reset_log()
        self.decay = decay
        self.debug = debug
        self.log = log

    @classmethod
    def reset_log(cls):
        try:
            os.remove(cls.log_filename)
        except OSError:
            pass

    def get_action(self, strategy):
        assert math.isclose(sum(strategy), 1, rel_tol=1e-4), "La strategia {strategy} non è valida"
        result = np.random.choice(self.actions, p=strategy)
        return result

    def get_strategy(self):
        # Get current mixed strategy through regret-matching
        decay_hist = np.array([self.decay**t for t in range(np.shape(self.regret_hist)[0])], ndmin=2).T
        regret_sum = np.sum(self.regret_hist*decay_hist[::-1], axis=0)
        normalizing_sum = 0
        strategy = np.zeros(3, dtype=float)
        for i in range(self.num_actions):
            strategy[i] = regret_sum[i] if regret_sum[i] > 0 else 0
            normalizing_sum += strategy[i]
        if normalizing_sum == 0:
            strategy = np.array([1.0/self.num_actions]*self.num_actions)
        else:
            strategy = strategy/normalizing_sum
        self.strategy_hist = np.concatenate((self.strategy_hist, np.array(strategy/sum(strategy), ndmin=2)))

        if self.log:
            with open(self.log_filename, 'a+') as fp:
                fp.write(', '.join(str(np.round(p, 4)) for p in strategy / sum(strategy)))
                fp.write("\n")

        return strategy

    def get_average_strategy(self):
        # Create a discount vector based on the decay rate. Its length is the same as the number of previous turns
        decay_hist = np.array([self.decay**t for t in range(np.shape(self.regret_hist)[0])], ndmin=2).T
        strategy_sum = np.sum(self.strategy_hist * decay_hist[::-1], axis=0)
        # Get average mixed strategy across all training iterations
        if sum(strategy_sum) > 0:
            return strategy_sum / sum(strategy_sum)
        else:
            return [1.0/self.num_actions]*self.num_actions

    def get_utility(self, a1, a2):
        assert all([a in self.actions for a in (a1, a2)]), f'Error while evaluating a1: {a1} vs a2: {a2}'
        if any([
            (a1 == 'R' and a2 == 'S'),
            (a1 == 'S' and a2 == 'P'),
            (a1 == 'P' and a2 == 'R'),
        ]):
            return 1
        elif a1 == a2:
            return 0
        else:
            return -1

    def batch_train(self, opp_actions):
        old_strategy = self.get_average_strategy()
        for opp_action in opp_actions:
            strategy = self.get_strategy()
            my_action = self.get_action(strategy)
            # Compute action utilities
            utilities = np.array([self.get_utility(a, opp_action) for a in self.actions])
            # Accumulate action regrets
            regret = utilities - np.array([self.get_utility(my_action, opp_action)] * self.num_actions)
            self.regret_hist = np.concatenate((self.regret_hist, np.array(regret, ndmin=2)))

            strategy_change = np.round(self.get_average_strategy() - old_strategy, 3)
            old_strategy = self.get_average_strategy()
            if self.debug:
                print(f"Opp. Action: {opp_action}, DeltaStrategy: {strategy_change}")

if __name__ == '__main__':

    model = CFRMinimizer(decay=1)

    # Batch training example
    epochs = 1000
    vs_strategy = np.array([0.4, 0.3, 0.3, ])  # Pr[R], Pr[P], Pr[S]
    vs_actions = [model.get_action(vs_strategy) for _ in range(epochs)]

    model.batch_train(vs_actions)

    # # Print training results
    x = np.arange(np.shape(model.strategy_hist)[0]-1)
    linestyles = ['dashed', 'dashdot', 'dotted']
    fig = plt.figure()
    for idx, action in enumerate(model.actions):
        plt.plot(x, np.cumsum(model.strategy_hist[1:, idx])/np.sum(np.cumsum(model.strategy_hist[1:], axis=0), axis=1),
                 label=action,
                 linestyle = linestyles[idx])
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Probabilities')
    plt.show()

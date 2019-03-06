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


class CFRMinimizer:

    log_filename = 'log.txt'

    def __init__(self, actions=('R', 'P', 'S', ), decay=1):
        self.num_actions = len(actions)
        self.actions = actions
        self.regret_hist = np.array([[0, 0, 0]], ndmin=2, dtype=float)
        self.strategy = np.zeros(shape=self.num_actions, dtype=float)
        self.strategy_hist = np.array(self.strategy, ndmin=2)
        self.reset_log()
        self.decay = decay

    @classmethod
    def reset_log(cls):
        try:
            os.remove(cls.log_filename)
        except OSError:
            pass

    def get_action(self, strategy):
        result = np.random.choice(self.actions, p=strategy)
        return result

    def get_strategy(self):
        # Get current mixed strategy through regret-matching
        decay_hist = np.array([self.decay**t for t in range(np.shape(self.regret_hist)[0])], ndmin=2).T
        regret_sum = np.sum(self.regret_hist*decay_hist, axis=0)
        normalizing_sum = 0
        for i in range(self.num_actions):
            self.strategy[i] = regret_sum[i] if regret_sum[i] > 0 else 0
            normalizing_sum += self.strategy[i]
        if normalizing_sum == 0:
            self.strategy = np.array([1.0/self.num_actions]*self.num_actions)
        else:
            self.strategy = self.strategy/normalizing_sum
        self.strategy_hist = np.concatenate((self.strategy_hist,
                                             np.array(self.strategy/sum(self.strategy), ndmin=2))
                                            )
        with open(self.log_filename, 'a+') as fp:
            fp.write(', '.join(str(np.round(p, 4)) for p in self.strategy / sum(self.strategy)))
            fp.write("\n")

        return self.strategy

    def get_average_strategy(self):
        decay_hist = np.array([self.decay ** t for t in range(np.shape(self.regret_hist)[0])], ndmin=2).T
        strategy_sum = np.sum(self.strategy_hist * decay_hist, axis=0)
        # Get average mixed strategy across all training iterations
        if sum(strategy_sum) > 0:
            return strategy_sum / sum(strategy_sum)
        else:
            return [1.0/self.num_actions]*self.num_actions

    @staticmethod
    def get_utility(a1, a2):
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

    def batch_train(self, num_epochs, opp_strategy):
        for _ in range(num_epochs):
            # Get regret-matched mixed-strategy actions
            self.get_strategy()
            my_action = self.get_action(self.strategy)  # 'R', 'P', 'S'
            opp_action = self.get_action(opp_strategy)  # 'R', 'P', 'S'
            # Compute action utilities
            utilities = np.array([self.get_utility(a, opp_action) for a in self.actions])
            # Accumulate action regrets
            regret = utilities - np.array([self.get_utility(my_action, opp_action)]*self.num_actions)
            self.regret_hist = np.concatenate((self.regret_hist, np.array(regret, ndmin=2)))

    def online_train(self, my_action, opp_action):
        assert my_action in self.actions, f"Errore nell'input di {self.online_train.__name__}: my_action={my_action}"
        assert opp_action in self.actions, f"Errore nell'input di {self.online_train.__name__}: opp_action={opp_action}"
        # Get regret-matched mixed-strategy actions
        self.get_strategy()
        # Compute action utilities
        utilities = np.array([self.get_utility(a, opp_action) for a in self.actions])
        # Accumulate action regrets
        regret = utilities - np.array([self.get_utility(my_action, opp_action)] * self.num_actions)
        self.regret_hist = np.concatenate((self.regret_hist, np.array(regret, ndmin=2)))

    def play(self):
        strategy = self.get_average_strategy()  # [0.2, 0.3, 0.5]
        return self.get_action(strategy)  # 'R', 'P', 'S'


if __name__ == '__main__':

    model = CFRMinimizer()

    # Batch training example
    epochs = 100
    vs_strategy = np.array([0.2, 0.2, 0.6, ])  # Pr[R], Pr[P], Pr[S]
    model.batch_train(epochs, vs_strategy)
    print(model.strategy_hist[:20])
    # Print training results
    x = np.arange(np.shape(model.strategy_hist)[0])
    fig = plt.figure()
    for idx, action in enumerate(model.actions):
        plt.plot(x, np.cumsum(model.strategy_hist[:, idx])/np.sum(np.cumsum(model.strategy_hist, axis=0), axis=1), label=action)
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Probabilities')
    plt.show()

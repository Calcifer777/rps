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

    action_labels = {0: 'R', 1: 'S', 2: 'P', 'R': 0, 'S': 1, 'P': 2}
    log_filename = 'log.txt'

    def __init__(self, actions=('R', 'P', 'S', )):
        self.num_actions = len(actions)
        self.actions = {k: v for k, v in zip(actions, range(self.num_actions))}
        self.regret_sum = np.zeros(shape=self.num_actions, dtype=float)
        self.strategy = np.zeros(shape=self.num_actions, dtype=float)
        self.strategy_hist = np.array(self.strategy, ndmin=2)
        self.strategy_sum = np.zeros(shape=self.num_actions, dtype=float)
        self.reset_log()

    @classmethod
    def reset_log(cls):
        try:
            os.remove(cls.log_filename)
        except OSError:
            pass

    def get_action(self, strategy):
        result = np.random.choice(list(self.actions.keys()), p=strategy)
        return result

    def get_strategy(self):
        # Get current mixed strategy through regret-matching
        normalizing_sum = 0
        for i in range(self.num_actions):
            self.strategy[i] = self.regret_sum[i] if self.regret_sum[i] > 0 else 0
            normalizing_sum += self.strategy[i]
        if normalizing_sum == 0:
            self.strategy = np.array([1.0/self.num_actions]*self.num_actions)
        else:
            self.strategy = self.strategy/normalizing_sum
        self.strategy_sum += self.strategy
        self.strategy_hist = np.concatenate((self.strategy_hist,
                                             np.array(self.strategy_sum/sum(self.strategy_sum), ndmin=2))
                                            )
        with open(self.log_filename, 'a+') as fp:
            for x in self.strategy_sum / sum(self.strategy_sum):
                fp.write(str(x))
            fp.write("\n")

        return self.strategy

    def get_average_strategy(self):
        # Get average mixed strategy across all training iterations
        if sum(self.strategy_sum) > 0:
            return self.strategy_sum / sum(self.strategy_sum)
        else:
            return [1.0/self.num_actions]*self.num_actions

    def get_utility(self, a1, a2):
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

    def batch_train(self, epochs, opp_strategy):
        for _ in range(epochs):
            # Get regret-matched mixed-strategy actions
            self.get_strategy()
            my_action = self.get_action(self.strategy)  # 0 | 1 | 2
            opp_action = self.get_action(opp_strategy)  # 0 | 1 | 2
            # Compute action utilities
            utilities = np.array([self.get_utility(action, opp_action) for action in self.actions.keys()])
            # Accumulate action regrets
            regret_update = utilities - np.array([self.get_utility(my_action, opp_action)]*self.num_actions)
            self.regret_sum += regret_update

    def online_train(self, my_action, opp_action):
        assert my_action in self.actions, f"Errore nell'input di {self.online_train.__name__}: my_action={my_action}"
        assert opp_action in self.actions, f"Errore nell'input di {self.online_train.__name__}: opp_action={opp_action}"
        # Get regret-matched mixed-strategy actions
        self.get_strategy()
        # Compute action utilities
        utilities = np.array([self.get_utility(action, opp_action) for action in self.actions.values()])
        # Accumulate action regrets
        regret_update = utilities - np.array([self.get_utility(my_action, opp_action)] * self.num_actions)
        self.regret_sum += regret_update

    def play(self):
        strategy = self.get_average_strategy()  # [0.2, 0.3, 0.5]
        return self.get_action(strategy)  # 'R', 'P', 'S'


if __name__ == '__main__':

    model = CFRMinimizer()

    # Batch training example
    epochs = 100
    vs_strategy = np.array([0.2, 0.2, 0.6,])  # Pr[R], Pr[P], Pr[S]
    model.batch_train(epochs, vs_strategy)
    print(model.strategy_hist[-5:, :])
    # Print training results
    x = np.arange(np.shape(model.strategy_hist)[0])
    fig = plt.figure()
    for k, v in model.actions.items():
        plt.plot(x, model.strategy_hist[:, v], label=k)
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Probabilities')
    plt.show()



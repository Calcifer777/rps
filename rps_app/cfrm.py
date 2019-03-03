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
from collections import namedtuple
from matplotlib import pyplot as plt


Score = namedtuple('Score', ['wins', 'losses', 'draws'])


class CFRMinimizer:

    def __init__(self):
        self.num_actions = 3
        self.actions = {'sasso': 0, 'carta': 1, 'forbici': 2, }
        self.regret_sum = np.zeros(shape=self.num_actions, dtype=float)
        self.strategy = np.zeros(shape=self.num_actions, dtype=float)
        self.strategy_hist = np.array(self.strategy, ndmin=2)
        self.strategy_sum = np.zeros(shape=self.num_actions, dtype=float)

    def get_action(self, strategy):
        result = np.random.choice(list(self.actions.keys()), p=strategy)
        return self.actions[result]

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
        return self.strategy

    def get_average_strategy(self):
        # Get average mixed strategy across all training iterations
        if sum(self.strategy_sum) > 0:
            return self.strategy_sum / sum(self.strategy_sum)
        else:
            return [1.0/self.num_actions]*self.num_actions

    def get_utility(self, action_1, action_2):
        result = (action_2 - action_1) % self.num_actions
        assert result in [0, 1, 2], 'Error in utilities computation'
        if result == 2:
            return 1
        elif result == 1:
            return -1
        else:
            return 0

    def batch_train(self, epochs, opp_strategy):
        for _ in range(epochs):
            # Get regret-matched mixed-strategy actions
            self.get_strategy()
            my_action = self.get_action(self.strategy)
            opp_action = self.get_action(opp_strategy)
            # Compute action utilities
            utilities = np.array([self.get_utility(action, opp_action)
                                  for action in self.actions.values()])
            # Accumulate action regrets
            self.regret_sum += utilities - np.array([self.get_utility(my_action, opp_action)]*self.num_actions)

    def online_train(self, my_action, opp_action):
        # Get regret-matched mixed-strategy actions
        self.get_strategy()
        # Compute action utilities
        utilities = np.array([self.get_utility(action, opp_action)
                              for action in self.actions.values()])
        # Accumulate action regrets
        self.regret_sum += utilities - np.array([self.get_utility(my_action, opp_action)] * self.num_actions)

    def update_score(self, score, my_action, player_action):
        utility = self.get_utility(player_action, my_action)
        if utility == 1:
            score['wins'] += 1
        elif utility == 0:
            score['draws'] += 1
        else:
            score['losses'] += 1
        return score

    def plot_training_results(self, fig):
        fig.clf()
        x = np.arange(np.shape(model.strategy_hist)[0])
        for k, v in self.actions.items():
            plt.plot(x, model.strategy_hist[:, v], label=k)
            plt.legend(loc='upper right')
            plt.xlabel('Turni')
            plt.ylabel('Probabilità')

    def play(self):
        score = {'wins': 0, 'draws': 0, 'losses': 0}

        while True:
            strategy = self.get_average_strategy()
            my_action = self.get_action(strategy)
            for k, v in self.actions.items():
                if my_action == v:
                    my_action_label = k
                    break
            else:
                assert False, 'Some error occurred'
            player_action_label = input('Fai la tua mossa (carta, forbice, sasso): ')
            if player_action_label in self.actions:
                player_action = self.actions[player_action_label]
            else:
                print("Mossa non valida.\n")
                continue
            print(f"{player_action_label} vs {my_action_label}")
            self.online_train(my_action, player_action)
            score = self.update_score(score, my_action, player_action)
            # Print updated results
            msg = f"V: {score['wins']}\tP: {score['draws']}\tS: {score['losses']}"
            print(msg)
            # print(self.strategy_hist)
            # self.plot_training_results(fig)

if __name__ == '__main__':

    model = CFRMinimizer()

    # # Batch training example
    # epochs = 100
    # vs_strategy = np.array([0.3, 0.4, 0.3,])
    # model.batch_train(epochs, vs_strategy)
    # # Print training results
    # x = np.arange(np.shape(model.strategy_hist)[0])
    # fig = plt.figure()
    # for k, v in model.actions.items():
    #     plt.plot(x, model.strategy_hist[:, v], label=k)
    # plt.legend(loc='upper right')
    # plt.xlabel('Epochs')
    # plt.ylabel('Probabilities')
    # plt.show()

    model.play()
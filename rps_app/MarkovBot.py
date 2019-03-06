import random
import itertools
from cfrm import CFRMinimizer


class MemoCfrm:

    def __init__(self, order=6, decay=0.7, actions=('R', 'P', 'S', )):
        self.actions = actions
        self.decay = decay
        self.order = order
        self.matrix = self.initialize_transition_probabilities(self.order)
        self.stm_seq = ''

    def initialize_transition_probabilities(self, order):
        keys = [''.join(i) for i in itertools.product(self.actions, repeat=order)]
        # Initializes the transition probabilities to the uniform distribution
        # p0 = {k: {'prob': 1.0 / 3, 'n_obs': 0, } for k in self.actions}
        transition_probabilities = {k: CFRMinimizer() for k in keys}
        return transition_probabilities

    def online_train(self, my_action, opp_action):
        flag_no_seq = len(self.stm_seq) < self.order
        self.stm_seq = self.stm_seq[1:] + input if len(self.stm_seq) == self.order else self.stm_seq + input
        if flag_no_seq:
            return
        self.matrix[self.stm_seq].online_training(my_action, opp_action)

    def play(self):
        if len(self.stm_seq) < self.order:
            return random.choice(['R', 'P', 'S'])
        else:
            return self.matrix[self.stm_seq].play()

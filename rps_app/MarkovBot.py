import random
import itertools
import math
from cfrm import CFRMinimizer


class MarkovChain:

    def __init__(self, order=6, decay=0.7, actions=('R', 'P', 'S', )):
        self.actions = actions
        self.decay = decay
        self.order = order
        self.matrix = self.initialize_transition_probabilities(self.order)
        self.stm_seq = ''

    def initialize_transition_probabilities(self, order):
        keys = [''.join(i) for i in itertools.product(self.actions, repeat=order)]
        # Initializes the transition probabilities to the uniform distribution
        p0 = {k: {'prob': 1.0 / 3, 'n_obs': 0, } for k in self.actions}
        transition_probabilities = {k: CFRMinimizer() for k in keys}
        return transition_probabilities

    def update_matrix(self, input):
        flag_no_seq = len(self.stm_seq) < self.order
        self.stm_seq = self.stm_seq[1:] + input if len(self.stm_seq) == self.order else self.stm_seq + input
        if flag_no_seq:
            return

        # Update obs number based on decay rate
        for i in self.matrix[self.stm_seq]:
            self.matrix[self.stm_seq][i]['n_obs'] = self.decay * self.matrix[self.stm_seq][i]['n_obs']
        # Update the obs number of last transition current state
        self.matrix[self.stm_seq][input]['n_obs'] = self.matrix[self.stm_seq][input]['n_obs'] + 1
        # Count the number of observation (in their decayed state) for the last transition
        n_total = 0
        for i in self.matrix[self.stm_seq]:
            n_total += self.matrix[self.stm_seq][i]['n_obs']
        # Normalize the probabilities so that they sum to one
        sum_prob = 0
        for i in self.matrix[self.stm_seq]:
            self.matrix[self.stm_seq][i]['prob'] = self.matrix[self.stm_seq][i]['n_obs'] / n_total
            sum_prob += self.matrix[self.stm_seq][i]['prob']
        assert math.isclose(sum_prob, 1.0, abs_tol=1e-4), 'Probabilities not normalized correctly'



    def predict(self):
        if len(self.stm_seq) < self.order:
            return random.choice(['R', 'P', 'S'])
        probs = {k: v['prob'] for k, v in self.matrix[self.stm_seq].items()}

        if max(probs.values()) == min(probs.values()):
            return random.choice(['R', 'P', 'S'])
        else:
            return max(probs, key=probs.get)

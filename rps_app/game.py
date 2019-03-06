from cfrm import CFRMinimizer
# from MarkovBot import MarkovChain


class Console:

    def __init__(self):
        self.actions = ('R', 'S', 'P')
        self.actions_dict = {'sasso': 'R', 'forbici': 'S', 'carta': 'P',
                             'R': 'sasso', 'S': 'forbici', 'P': 'carta', }
        self.print_welcome_msg()
        self.ai = CFRMinimizer(actions=self.actions)
        self.score = {'wins': 0, 'draws': 0, 'losses': 0, }

    @staticmethod
    def print_welcome_msg():
        print("\n\nCIAO, PRONTO A GIOCARE?\n\n")

    def get_player_action(self):
        player_action_label = input('\nFai la tua mossa (carta, forbici, sasso): ')
        while True:
            if player_action_label in self.actions_dict:
                player_action = self.actions_dict[player_action_label]
                break
            else:
                print("Mossa non valida.\n")
        return self.actions_dict[player_action]

    def update_score(self, player_action, ai_action, ):
        if any([
            (ai_action == 'R' and player_action == 'P'),
            (ai_action == 'S' and player_action == 'R'),
            (ai_action == 'P' and player_action == 'S'),
        ]):
            self.score['wins'] += 1
        elif ai_action == player_action:
            self.score['draws'] += 1
        else:
            self.score['losses'] += 1

    def play(self, total_turns=100):
        turns = 1
        for _ in range(total_turns):
            ai_action = self.ai.play()
            player_action = self.get_player_action()
            self.ai.online_train(ai_action, self.actions_dict[player_action])
            self.update_score(self.actions_dict[player_action], ai_action)
            print(f"{player_action} vs {self.actions_dict[ai_action]}")
            # Print updated results
            msg = (f"G: {turns}\t"
                   f"V: {self.score['wins']}\t"
                   f"P: {self.score['draws']}\t"
                   f"S: {self.score['losses']}")
            print(msg)
            turns += 1


if __name__ == '__main__':

    console = Console()

    console.play()
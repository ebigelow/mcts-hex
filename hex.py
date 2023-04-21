"""
Hex game, with automation

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from mcts import Mcts
from minihex import *

class HexState:
    def __init__(self, sim, player=1) -> None:
        self.sim = sim
        self.player = player
    
    def flip(self):
        new_state = HexState(self.sim.copy(), 1 - self.player)
        return new_state
    
    def getCurrentPlayer(self):
        if self.sim.active_player == self.player:
            return 1
        else:
            return -1
    
    def getPossibleActions(self):
        return self.sim.get_possible_actions()

    def takeAction(self, action):
        sim = self.sim.copy()
        sim.make_move(action)
        return HexState(sim, player=self.player)
    
    def isTerminal(self):
        return self.sim.done

    def getReward(self):
        if self.sim.winner == self.player:
            return 1
        elif self.sim.winner == 1 - self.player:
            return -1
        else:
            return 0


def make_init(board_size=5, first_player=1):
    sim = HexGame(first_player, player.EMPTY * np.ones((board_size, board_size)))
    return HexState(sim)


def run_game(board_size=5, iters=100):
    init_state = make_init(board_size=board_size, first_player=1)

    transcript = []

    player_1 = Mcts(iterationLimit=iters)
    player_0 = Mcts(iterationLimit=iters)

    p1_mv = player_1.start(init_state)['action']
    next_state = player_1.root.children[p1_mv].state.flip()
    p0_mv = player_0.start(next_state)['action']

    transcript.extend([p1_mv, p0_mv])

    # print_board(player_0.root.children[p0_mv].state.sim.board)

    while True:
        p1_mv = player_1.consume_action(p1_mv, p0_mv)['action']
        transcript.append(p1_mv)

        if player_1.root.children[p1_mv].isTerminal:
            break

        p0_mv = player_0.consume_action(p0_mv, p1_mv)['action']
        transcript.append(p0_mv)

        # print_board(player_0.root.children[p0_mv].state.sim.board)
        if player_0.root.children[p0_mv].isTerminal:
            break

    return transcript

def run_game_proc(kwargs):
    return run_game(**kwargs)


if __name__ == '__main__':
    n_games = 10000
    n_cores = 12
    board_size = 5
    mcts_iters = 100

    with Pool(n_cores) as pool:
        kwargs = dict(board_size=board_size, iters=mcts_iters)

        ts = list(tqdm(
            pool.imap(run_game_proc, [kwargs] * n_games), total=n_games))

    
    out = [
        {
            'mcts_iters': mcts_iters,
            'board_size': board_size,
            'transcripts': ts
        }
    ]

    df = pd.DataFrame(out)
    df.to_json(f'hex_data-{board_size}.json')




# <codecell>

'''

init_state = make_init(5)
searcher = Mcts(iterationLimit=1000)
results = searcher.start(init_state)
print('results', results)

while not searcher.root.state.isTerminal():
    a = results['action']

    print_board(searcher.root.children[a].state.sim.board)

    opp_a = input('your move: ')
    opp_a = int(opp_a)

    results = searcher.consume_action(a, opp_a)
    print('results', results)



# <codecell>

env = HexEnv(opponent_policy=random_policy, board_size=3)

state, info = env.reset()
done = False
while not done:
    board, player = state
    action = random_policy(board, player, info)
    state, reward, done, info = env.step(action)

env.render()

if reward == -1:
    print("Player (Black) Lost")
elif reward == 1:
    print("Player (Black) Won")
else:
    print("Draw")
'''

"""
Hex game, with automation

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
from minihex import HexEnv, random_policy

env = HexEnv(opponent_policy=random_policy, board_size=11)

state, info = env.reset()
done = False
while not done:
    board, player = state
    action = random_policy(board, player, info)
    print('ACTION', action)
    state, reward, done, info = env.step(action)

env.render()

if reward == -1:
    print("Player (Black) Lost")
elif reward == 1:
    print("Player (Black) Won")
else:
    print("Draw")

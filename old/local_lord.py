import localagents
import hangman
from tqdm import tqdm
import logging
import sys

logging.basicConfig(level=logging.DEBUG, filename='stats.log')
# logging.debug('This is a debug message')
# logging.info('This is an info message')
# logging.warning('This is a warning message')
#! getting statistics for the basic strategy
def get_strats(agent, t = 0.55, n = 100):
    api = agent(threshold=t)
    won = 0
    played = 0
    for _ in (pbar := tqdm(range(n))):
        ans = api.start_game(practice=1,verbose=False)
        if (ans): won += 1
        played += 1
        pbar.set_description("Acc. %.3f"%(won/played))
    logging.info(f"Accuracy with {agent} {won/played : .3f}")


# get_strats(localagents.LocalHangman)
# get_strats(localagents.localGreedy)
# get_strats(localagents.ngramSolver, 0.55)

get_strats(localagents.ngramSolver, 0.50)

# get_strats(localagents.ngramSolver, 0.45)
if __name__ == '__main__':
    print(f"Solving for {sys.argv[1]}")
    get_strats(localagents.ngramSolver, eval(sys.argv[1]))
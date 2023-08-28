
import time
from tqdm import tqdm

from agents import ngramSolver as betterHangman

api = betterHangman(access_token="a9329fd58d47d399d21ba1a9a7f8e1", timeout=2000)
api.hangman_url = 'https://trexsim.com/trexsim/hangman'





import dummylog
from datetime import datetime

dl = dummylog.DummyLog(datetime.now().strftime("%d%m_%H%M%S"))


def gen_stats(n):
    won = 0
    played = 0
    for _ in (pbar := tqdm(range(n))):
        ans = api.start_game(practice=1,verbose=False)
        if (ans): won += 1
        played += 1
        [total_practice_runs,total_recorded_runs,total_recorded_successes,total_practice_successes] = api.my_status() # Get my game stats: (# of tries, # of wins)
        # practice_success_rate = total_practice_successes / total_practice_runs
        pbar.set_description("Acc. %.3f"%(won/played))
        time.sleep(0.5)
    dl.logger.info(f"After {played} games, won {won/played : .3f} games")
    print("Used up %d games practice games out of an allotted 100,000" %total_practice_runs)


import sys
if __name__ == '__main__':
    print(sys.argv)
    gen_stats(eval(sys.argv[1]))
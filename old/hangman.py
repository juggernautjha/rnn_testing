import numpy as np
from typing import List, Tuple, Dict, Union
# requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
import random

#! Barebones Python implementation of Hangman, fuelled by 2 bottles of Sting and your mom.
class Hangman():
    '''
    Class for the hangman game, initialized using dictionary, and randomly picked word. 
    '''
    def __init__(self, word_list : List[str], max_tries: int = 6):
        self.word_list = [i.strip() for i in open(word_list).readlines()]
        random.shuffle(self.word_list)
        random_idx = np.random.randint(0, len(word_list))
        self.word = self.word_list[random_idx]
        # print(self.word_list[:5])
        self.length = len(self.word)
        self.state = '_'*(self.length)
        self.max_tries = max_tries
        self.curr_tries = 0
        self.status = 'ongoing'
        self.game_id = f'6969rahuljha_{random.randint(0, 1313453)}'

    def evaluate_guess(self, guess: str) -> Dict:
        '''
        Compares the guess with the hidden word, and returns the state after changing it.
        Explanation: suppose the word is RAMESH and the initial state is ______. I guess HUMKEH, the state be3comes __M__H, which is what
        we return. 
        Will probably use it to filter the dictionary.
        '''
        # assert self.length == len(guess), "Invalid Guess" #pretty my agent eliminates the need for this check, but I was writing a generic ass hangman playe        
        if guess not in self.word:
            self.curr_tries += 1
        
        self.state = ''.join([guess if guess == self.word[i] else self.state[i] for i in range(self.length)]) #! correct positions persistent states.
        if (self.word == self.state):
            self.status = 'success'
        elif (self.curr_tries == self.max_tries):
            self.status = 'failed'
        return {
            'game_id' : self.game_id,
            'status' : self.status,
            'tries_remains' : self.max_tries - self.curr_tries,
            'word' : self.state
        }
    

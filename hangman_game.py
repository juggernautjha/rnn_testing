import numpy as np
from typing import List, Tuple, Dict, Union
# requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
import random
from base import Agent
import re
import collections
from tqdm import tqdm
#! Barebones Python implementation of Hangman, fuelled by 2 bottles of Sting and your mom.
class Hangman():
    '''
    Class for the hangman game, initialized using dictionary, and randomly picked word. 
    '''
    def __init__(self, word_list : str = "data/250k_complement.txt", max_tries: int = 6):
        self.word_list = [i.strip() for i in open(word_list).readlines()]
        random.shuffle(self.word_list)
        random_idx = np.random.randint(0, len(word_list))
        self.word = self.word_list[random_idx]
        self.length = len(self.word)
        self.state = '_'*(self.length)
        self.max_tries = max_tries
        self.curr_tries = 0
        self.status = 'ongoing'
        self.game_id = f'6969rahuljha_{random.randint(0, 1313453)}'

    def evaluate_guess(self, guess: str) -> Dict:
        '''
        Compares the guess with the hidden word, and returns the state after changing it.
        Explanation: suppose the word is RAMESH and the initial state is ______. I guess HUMKEH, the state becomes __M__H, which is what
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
    


class LocalAgent(Agent):
    def __init__(self, access_token=None, session=None, timeout=None, full_dictionary_location="data/250k.txt"):
        super().__init__(access_token, session, timeout, full_dictionary_location)
    

    def guess(self, word): # word input example: "_ p p _ e "
        ###############################################
        # Replace with your own "guess" function here #
        ###############################################

        # clean the word so that we strip away the space characters
        # replace "_" with "." as "." indicates any character in regular expressions
        clean_word = word[::2].replace("_",".")
        
        # find length of passed word
        len_word = len(clean_word)
        
        # grab current dictionary of possible words from self object, initialize new possible words dictionary to empty
        current_dictionary = self.current_dictionary
        new_dictionary = []
        
        # iterate through all of the words in the old plausible dictionary
        for dict_word in current_dictionary:
            # continue if the word is not of the appropriate length
            if len(dict_word) != len_word:
                continue
                
            # if dictionary word is a possible match then add it to the current dictionary
            if re.match(clean_word,dict_word):
                new_dictionary.append(dict_word)
        
        # overwrite old possible words dictionary with updated version
        self.current_dictionary = new_dictionary
        
        
        # count occurrence of all characters in possible word matches
        full_dict_string = "".join(new_dictionary)
        
        c = collections.Counter(full_dict_string)
        sorted_letter_count = c.most_common()                   
        
        guess_letter = '!'
        
        # return most frequently occurring letter in all possible words that hasn't been guessed yet
        for letter,instance_count in sorted_letter_count:
            if letter not in self.guessed_letters:
                guess_letter = letter
                break
            
        # if no word matches in training dictionary, default back to ordering of full dictionary
        if guess_letter == '!':
            sorted_letter_count = self.full_dictionary_common_letter_sorted
            for letter,instance_count in sorted_letter_count:
                if letter not in self.guessed_letters:
                    guess_letter = letter
                    break            
        
        return guess_letter



    def start_game(self, practice=True, verbose=True):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary        

        game = Hangman()
        game_id = game.game_id
        word = game.state
        tries_remains = game.max_tries
        if verbose:
            print("Successfully start a new game! Game ID: {0}. # of tries remaining: {1}. Word: {2}.".format(game_id, tries_remains, word))
        while tries_remains>0:
            # get guessed letter from user code
            guess_letter = self.guess(word)
                
            # append guessed letter to guessed letters field in hangman object
            self.guessed_letters.append(guess_letter)
            if verbose:
                print("Guessing letter: {0}".format(guess_letter))
                
            try:    
                res = game.evaluate_guess(guess_letter)
            except Exception as e:
                print('Other exception caught on request.')
                raise e
            
            if verbose:
                print("Sever response: {0}".format(res))
            status = res['status']
            tries_remains = res['tries_remains']
            if status=="success":
                if verbose:
                    print("Successfully finished game: {0}".format(game_id))
                return True
            elif status=="failed":
                reason = res.get('reason', '# of tries exceeded!')
                if verbose:
                    print("Failed game: {0}. Because of: {1}".format(game_id, reason))
                return False
            elif status=="ongoing":
                word = res['word']
        return status=="success"        

if __name__ == '__main__':
    agent = LocalAgent()
    win = 0
    perc = 0
    for i in (pbar := tqdm(range(1000))):
        if (agent.start_game(verbose=False)):
            win += 1
            perc = win/(i+1)
        pbar.set_description(f"%.3f" %perc)
    
    print(win)


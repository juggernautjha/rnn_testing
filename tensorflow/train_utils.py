import typing
from model import KerasModel as Model
import numpy as np
import random
CHAR_TO_IDX = {
    chr(i+97) : i+1 for i in range(26)
}
CHAR_TO_IDX['_'] = 27
IDX_TO_CHAR = {CHAR_TO_IDX[i] : i for i in CHAR_TO_IDX}
MAXLEN = 30

from tqdm import tqdm

#! This is the tensorflow implementation of my RNN model.
def pad_tensor(t : typing.List, length : int = MAXLEN):
    if len(t) > length:
        return t[-len::1]
    else:
        return [0]*(MAXLEN - len(t)) + t

class Trainer:
    '''
    Class to create training data.
    Parameters:
        word : string -> used to create training samples.
        guessing_model : torch.nn.model -> model that actually learns. 
        tries: int -> self explanatory, maximum number of tries during training.
        verbose: bool -> for debugging.
    
        
    Functions:
        get_game_state(self) -> List[typing.List] -> returns both the padded gamestate and the one-hot encoded matrix of gamestate.
        get_evaluation_tensors(self) -> typing.List[typing.List] -> returns the one-hot encoded guess and the expected probability vector.
    '''
    def __init__(self, word : str, guessing_model, tries : int = 6, verbose : bool = False):
        self.word = word
        self.word_rep = [CHAR_TO_IDX[i] for i in word]
        self.guessed = set([])
        self.remaining = set(self.word_rep)
        self.tries_remain = tries
        self.game_state = [27 for i in word]
        self.training_set = {
            'game_state' : [],
            'game_state_one_hot' : [],
            'guessed_one_hot' : [], #! previously guessed letters.
            'guessed' : [],
            'expected_letters' : [] #! list of probability vectors 
        }
        self.guessing_model = guessing_model
        self.verbosity = verbose

    def get_game_state(self) -> typing.List[typing.List]:
        '''
        One-hot encoded vector corresponding to each character.
        Verified, works.
        '''
        game_state = [i if i in self.guessed else 27 for i in self.word_rep]
        game_state_one_hot = np.eye(28)[game_state]
        # game_state_one_hot = F.one_hot(x = game_state, num_classes=28)
        return game_state, game_state_one_hot

    def get_evaluation_tensors(self) -> typing.List[typing.List]:
        '''
        Returns the one-hot encoded guess tensor, and returns the expected guesses at this stage.
        Verified, works. 
        '''
        guessed = [1 if i+1 in self.guessed else 0 for i in range(26)]
        answer = [1 if i+1 in self.remaining else 0 for i in range(26)]
        s = sum(answer)
        answer = [i/s for i in answer]
        # answer = answer/answer.sum()
        return guessed, answer
    
    def action_replay(self) -> None:
        '''
        Sorry for the Cheesy-ass name. Basically replays the game. 
        '''
        statistics = self.training_set
        for i in range(1, len(statistics['game_state_one_hot'])):
            print(f"Saw: {statistics['game_state'][i]}, Guessed: {statistics['guessed'][i]}")
        return

    def update_statistics(self, guess : int) -> None:
        '''
        At the end of each game, update the statistics dictionary.
        '''
        if guess == -1:
            #! initial initialization. I am stupid.
            game_state, game_state_one_hot = self.get_game_state()
            guessed_one_hot, answer = self.get_evaluation_tensors()
            game_state = pad_tensor(game_state)
            #! Add 'things' to the training set. 
            self.training_set['game_state'].append(game_state)
            self.training_set['game_state_one_hot'].append(game_state_one_hot)
            self.training_set['guessed_one_hot'].append(guessed_one_hot)
            self.training_set['expected_letters'].append(answer)
            return
        #! Reasonably obvious?
        self.guessed.add(guess)
        game_state, game_state_one_hot = self.get_game_state()
        guessed_one_hot, answer = self.get_evaluation_tensors()
        game_state = pad_tensor(game_state)
        #! Add 'things' to the training set. 
        self.training_set['game_state'].append(game_state)
        self.training_set['game_state_one_hot'].append(game_state_one_hot)
        self.training_set['guessed_one_hot'].append(guessed_one_hot)
        self.training_set['expected_letters'].append(answer)
        self.training_set['guessed'].append(guess)
        if guess in self.remaining:
            self.remaining.remove(guess)
        else:
            self.tries_remain -= 1
        return

    def show_game_board(self):
        '''
        Utility function for showing the gameboard. 
        '''
        print(''.join([IDX_TO_CHAR[i] for i in self.game_state]))


    def get_guess_vector(self):
        '''
        Utility function to evaluate guessing model on the last game state and latest guessed vector. 
        Returns an np.array of probabilities.
        '''
        game_state = np.vstack(self.training_set['game_state'][-1:]).astype(float)
        guessed = np.vstack(self.training_set['guessed_one_hot'][-1:]).astype(float)
        probab_vector = self.guessing_model(game_state, guessed)
        return probab_vector

    def play(self) -> typing.Tuple:
        '''
        Makeshift function for debugging. The true trainer class will inherit from this and
        overwrite this function. 
        '''
        correct_guesses = 0
        wrong_guesses = 0
        self.update_statistics(-1)
        if self.verbosity: self.show_game_board()
        if (self.verbosity): print("Now Play")
        while len(self.remaining) > 0 and self.tries_remain > 0:
            if (self.verbosity): print(f"Tries remaining : {self.tries_remain}")
            
            probab_vector = self.get_guess_vector()
            #! greedy
            char_idx = np.argmax(probab_vector)
            guess = IDX_TO_CHAR[char_idx+1]
            if guess in self.word: correct_guesses += 1
            else: wrong_guesses += 1
            if (self.verbosity): print(guess)
            self.update_statistics(CHAR_TO_IDX[guess])
            if (self.verbosity): print([IDX_TO_CHAR[i] for i in self.guessed])
            self.game_state = [i if i in self.guessed else 27 for i in self.word_rep]
            if (self.verbosity): print(self.game_state)
            # self.show_game_board()
        if (self.verbosity): print(self.word)     
        return correct_guesses, wrong_guesses, self.tries_remain != 0      


        
    def game_stats(self):
        status = self.tries_remain != 0
        return status  

class Train_on_Batch():
    '''
    True trainer, takes in a word source, and the number of games. At the end of those many games it trains the model, 
    evaluates and hopefully does well on the final evaluation.
    '''
    def __init__(self, word_list: str, guessing_model, num_games : int = 10, tries: int = 6, verbose: bool = True):
        # super().__init__(word, guessing_model, tries, verbose)
        self.words = [i.strip() for i in open(word_list).readlines()]
        self.guessing_model = guessing_model
        self.num_games = num_games
        self.tries = tries
        self.verbose = verbose
        self.batch_memory = {
            'game_state' : [],
            'game_state_one_hot' : [],
            'guessed_one_hot' : [], #! previously guessed letters.
            'guessed' : [],
            'expected_letters' : [] 
        }
    

    def pick_random_word(self):
        return random.choice(self.words)

    def train_one_batch(self, epochs:int = 10):
        '''
        Using Keras as opposed to pytorch gives us the advantage of training on slightly
        different batch sizes (as in I would expect the dataset size for 10 games to be roughly the same across games.)
        '''
        #! Generating data for training
        for i in range(self.num_games):
            word =self.pick_random_word()
            player = Trainer(word, self.guessing_model, self.tries, self.verbose)
            player.play()
            for key in self.batch_memory:
                self.batch_memory[key] += player.training_set[key]

        #! Actually training
        game_states = np.vstack(self.batch_memory['game_state']).astype(float)
        guessed = np.vstack(self.batch_memory['guessed_one_hot']).astype(float)
        expected = np.vstack(self.batch_memory['expected_letters']).astype(float)
        assert len(game_states) == len(expected)
        assert len(guessed) == len(game_states)
        for i in tqdm(range(epochs)):
            self.guessing_model.guessing_model.train_on_batch([game_states, guessed], expected)
    
    def reset(self):
        '''
        Resets the batch_memory        
        '''
        self.batch_memory = {
            'game_state' : [],
            'game_state_one_hot' : [],
            'guessed_one_hot' : [], #! previously guessed letters.
            'guessed' : [],
            'expected_letters' : [] 
        }

if __name__ == '__main__':
    # n = Trainer("rahul", None, 7)
    # n.guessed = [17]
    # print(n.remaining)
    # print(n.get_game_state())
    # # print(n.get_guessed_onehot(17))
    model = Model("base_config")
    wl = "/home/juggernautjha/Desktop/trexquant/rnn_testing/data/250k_complement.txt"
    trainer = Train_on_Batch(wl, model, 1)
    trainer.train_one_batch(5)

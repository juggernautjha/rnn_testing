import torch
import torch.nn.functional as F
import typing
from torch import tensor as T
from model import Model
import numpy as np
import random
CHAR_TO_IDX = {
    chr(i+97) : i+1 for i in range(26)
}
CHAR_TO_IDX['_'] = 27
IDX_TO_CHAR = {CHAR_TO_IDX[i] : i for i in CHAR_TO_IDX}
MAXLEN = 30

def pad_tensor(t : torch.tensor, length : int = MAXLEN):
    if len(t) > length:
        return t[-len::1]
    else:
        return F.pad(t, (MAXLEN - len(t), 0), value = 0)



class Trainer:
    '''
    Class to create training data.
    Parameters:
        word : string -> used to create training samples.
        guessing_model : torch.nn.model -> model that actually learns. 
        tries: int -> self explanatory, maximum number of tries during training.
        verbose: bool -> for debugging.
    
        
    Functions:
        get_game_state(self) -> List[torch.tensor] -> returns both the padded gamestate and the one-hot encoded matrix of gamestate.
        get_evaluation_tensors(self) -> typing.List[torch.tensor] -> returns the one-hot encoded guess and the expected probability vector.
    '''
    def __init__(self, word : str, guessing_model, tries : int = 6, verbose : bool = True):
        self.word = word
        self.word_rep = [CHAR_TO_IDX[i] for i in word]
        self.guessed = set([])
        self.remaining = set(self.word_rep)
        self.tries_remain = tries
        self.game_state = torch.tensor([27 for i in word])
        self.training_set = {
            'game_state' : [],
            'game_state_one_hot' : [],
            'guessed_one_hot' : [], #! previously guessed letters.
            'guessed' : [],
            'expected_letters' : [] #! list of probability vectors 
        }
        self.guessing_model = guessing_model
        self.verbosity = verbose

    def get_game_state(self) -> typing.List[torch.tensor]:
        '''
        One-hot encoded vector corresponding to each character.
        Verified, works.
        '''
        game_state = torch.tensor([i if i in self.guessed else 27 for i in self.word_rep])
        game_state_one_hot = F.one_hot(x = game_state, num_classes=28)
        return game_state, game_state_one_hot

    def get_evaluation_tensors(self) -> typing.List[torch.tensor]:
        '''
        Returns the one-hot encoded guess tensor, and returns the expected guesses at this stage.
        Verified, works. 
        '''
        guessed = torch.tensor([1 if i+1 in self.guessed else 0 for i in range(26)], dtype=torch.float)
        answer = torch.tensor([1 if i+1 in self.remaining else 0 for i in range(26)], dtype=torch.float)
        answer = answer/answer.sum()
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
        print(''.join([IDX_TO_CHAR[i.item()] for i in self.game_state]))

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
            game_state = self.training_set['game_state'][-1:]
            guessed = self.training_set['guessed_one_hot'][-1:]
            probab_vector = self.guessing_model(torch.stack(game_state), torch.stack(guessed))[0][0]
            #! greedy
            for i in range(len(guessed[0])):
                if guessed[0][i] == 1:
                    probab_vector[i] = 0.0
            char_idx = torch.argmax(probab_vector).item()
            guess = IDX_TO_CHAR[char_idx+1]
            if guess in self.word: correct_guesses += 1
            else: wrong_guesses += 1
            if (self.verbosity): print(guess)
            self.update_statistics(CHAR_TO_IDX[guess])
            if (self.verbosity): print([IDX_TO_CHAR[i] for i in self.guessed])
            self.game_state = torch.tensor([i if i in self.guessed else 27 for i in self.word_rep])
            if (self.verbosity): print(self.game_state)
            # self.show_game_board()
        if (self.verbosity): print(self.word)     
        return correct_guesses, wrong_guesses, self.tries_remain != 0      
    def game_stats(self):
        status = self.tries_remain != 0
        return status  
    
    @staticmethod
    def call_model(model : torch.nn.Module, param_1 : typing.List, param_2 : typing.List) -> T:
        p = np.array([param_1])
        q = np.array([param_2]).astype(float)
        p = T(p)
        q = T(q)
        return model(p, q)

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
    
    def generate_data(self):
        '''
        Generates data for training. 
        '''
        for i in range(self.num_games):
            word = random.randint(0, len(self.words)-1)
            word = self.words[word]
            player = Trainer(word, self.guessing_model, self.tries, self.verbose)
            player.play()
            for j in player.training_set:
                self.batch_memory[j] += player.training_set[j]

        
        # self.batch_memory = {
        #     i : torch.stack(self.batch_memory[i]) for i in self.batch_memory if i != 'game_state_one_hot'
        # }

        game_state = torch.stack(self.batch_memory['game_state'])
        guessed = torch.stack(self.batch_memory['guessed_one_hot'])
        expected = torch.stack(self.batch_memory['expected_letters'])
        return game_state, guessed, expected



class Evaluator(Train_on_Batch):
    '''
    Likely overkill but evaluates the model after each epoch. The way it does it is:
        Play N games.
        Keeps track of:
            - Correct Guesses/Game
            - Wrong Guesses/Game
            - Games Won      
    '''
    def __init__(self, word_list: str, guessing_model, num_games: int = 10, tries: int = 6, verbose: bool = True):
        super().__init__(word_list, guessing_model, num_games, tries, verbose)

    def evaluate(self):
        correct = 0
        incorrect = 0
        games_won = 0
        for i in range(self.num_games):
            word = random.randint(0, len(self.words)-1)
            word = self.words[word]
            player = Trainer(word, self.guessing_model, self.tries, self.verbose)
            c, ic, gw = player.play()
            correct += c
            incorrect += ic
            if gw: games_won += 1
        return correct/self.num_games, incorrect/self.num_games, games_won/self.num_games

    
if __name__ == '__main__':
    # n = Trainer("rahul", None, 7)
    # n.guessed = [17]
    # print(n.remaining)
    # print(n.get_game_state())
    # # print(n.get_guessed_onehot(17))
    model = Model("base_config")
    z = Train_on_Batch("data/250k.txt", model)
    gs, goh, el = z.generate_data()
   
    print(gs.size(), goh.size(), el.size())

    assert(len(gs) ==  len(goh))
    assert(len(goh) ==  len(el))

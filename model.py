from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from torch.nn import Embedding, Linear, LSTM, Module

from torch import tensor as T

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

import lightning as L
from torch.utils.data import TensorDataset, DataLoader


import matplotlib.pyplot as plt
import seaborn as sns

import typing

from tqdm.notebook import tqdm

import yaml
'''
configs directory: contains yaml of hyperparameters for quick testing.
'''
CONFIGS = "configs"

class HangmanTrainingDataset(Dataset):
    """
    Dataset for training. 
    Params:
        training_set : Dictionary defined in train.py. 
        training_set = {
            'game_state' : [],
            'game_state_one_hot' : [],
            'guessed_one_hot' : [], #! previously guessed letters.
            'guessed' : [],
            'expected_letters' : [] #! list of probability vectors 
        }
    """
    def __init__(self, game_state, guessed_one_hot, expected) -> None:
        super().__init__()
        self.game_states = game_state
        self.guessed_one_hot = guessed_one_hot
        self.expected = expected

    def __len__(self) -> None:
        assert len(self.game_states) == len(self.guessed_one_hot), "Assertion Failed"
        assert len(self.game_states) == len(self.expected)
        return len(self.game_states)

    def __getitem__(self, index) -> typing.Any:
        # return super().__getitem__(index)
        return (self.game_states[index], self.guessed_one_hot[index]), self.expected[index]
    

class Model(nn.Module):
    """
    Model implementation. This implementation uses embeddings.
    Params:
        config_file : str
            Location of the configuration file.
        config_file contains:
            vocab_size: number of unique characters. 27 in our case.
            embedding_dim: dimension of embeddings.
            n_lstm: number of LSTM layers.
            hidden_dim: LSTM hidden dimension.
            dense_dim_lstm: LSTM dense dimension.
            dense_dim_guessed: neurons in the dense layer.
            n_dense_guessed: number of dense layers for guessed vector.
            dense_dim_final: dimension of final dense layer
            dense_dim_output: number of output neurons. 26 in our case.
    """
    def __init__(self, config: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        stream = open(f"{CONFIGS}/{config}.yaml", "r").read()
        self.config = yaml.load(stream, Loader=yaml.Loader)
        # config = dict(self.config)
        d = []
        for i in self.config:
            d.append(list(i.items())[0])
        self.config = dict(d)
        config = self.config
        #! DEFINING LSTM
        '''
        Game State -> Embedding -> n_lstm number of LSTM layers
        -> dense layer
        '''
        self.embedding = Embedding(config['vocab_size'] + 1, 
                                   config['embedding_dim'], 
                                   padding_idx=0)
        self.lstm = LSTM(config['embedding_dim'], config['hidden_dim'],num_layers = 2, bidirectional=True, batch_first = True)
        self.lstm_dense = nn.Linear(config['hidden_dim']*2, config['dense_dim_lstm'])


        #! DEFINING GUESSED VECTOR ENCODING
        '''
        guessed_one_hot(26, dense_dim_guessed) -> n_dense_guessed dense layers
        '''
        self.guessed_dense = nn.Sequential(
            nn.Linear(26, config['dense_dim_guessed']),
            nn.ReLU(),
            nn.Linear(config['dense_dim_guessed'], config['dense_dim_guessed']),
            nn.ReLU()
        )

        #! Final softmax yayy
        self.final_dense = nn.Sequential(
            nn.Linear(config['hidden_dim']*4, config['dense_dim_final']),
            nn.ReLU(),
            nn.Linear(config['dense_dim_final'], config['dense_dim_output']),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self, game_state : torch.tensor, guessed : torch.tensor):
        '''
        Forward pass.
        game_state : torch.Tensor. 
                    shape: (batch_size, max_len)
        guessed : torch.Tensor.
                    shape: (batch_size, vocab_size)
        '''
        game_state = self.embedding(game_state)
        game_state, (h_T, c_T) = self.lstm(game_state)
        states = []
        game_state = torch.mean(game_state, 1, keepdim=True)
        for i in game_state:
            states.append(i)
        game_state = torch.cat(tuple(states), 0)
        game_state = self.lstm_dense(game_state)
        game_state = F.relu(game_state)
        
        guessed = self.guessed_dense(guessed)


        concat = torch.cat((game_state, guessed), 1)
        answer = self.final_dense(concat)
        return answer, game_state, guessed
    

    



# trainer = L.Trainer()
# trainer.fit(m, )
    
    
    

def compute_loss(model, dataset):
    model.eval()
    all_losses = []
    for x, y in dataset:
        res = model(x[0], x[1])
        loss = F.cross_entropy(res, y)
        all_losses.append(loss)
    return all_losses


def train_one_batch(model, game_state, guessed, expected, epochs, optimizer):
    '''
    Trains model for batch.
    Parameters:
        model : torch.nn.Module -> model that needs to be trained.
        game_state : torch.Tensor -> tensor of game_states. 
        guessed : torch.Tensor -> tensor of one hot encoded tensors of guessed states.
        expected : torch.Tensor -> tensor of expected prbabilities. 
    '''
    pbar = tqdm(range(epochs))
    for i in pbar:
        print(i)
        pred, _ , _ = model(game_state, guessed)
        loss = F.binary_cross_entropy(pred, expected)
        pbar.set_description("%.3f loss" % loss)
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        



    





if __name__ == '__main__':
    game_state = [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0, 27,  1, 27, 27, 27,  9, 27, 27, 27, 27]]
    guessed = [[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0]]
    # correct = [[0.1429, 0.0000, 0.0000, 0.1429, 0.0000, 0.0000, 0.1429, 0.0000, 0.1429,
    #     0.0000, 0.0000, 0.1429, 0.0000, 0.1429, 0.0000, 0.0000, 0.0000, 0.0000,
    #     0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1429, 0.0000],
    #             [0.0000, 0.0000, 0.0000, 0.2000, 0.0000, 0.0000, 0.2000, 0.0000, 0.2000,
    #     0.0000, 0.0000, 0.1429, 0.0000, 0.1429, 0.0000, 0.0000, 0.0000, 0.0000,
    #     0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2000, 0.0000]]
    
    # gs = HangmanTrainingDataset(T(game_state), T(guessed, dtype=torch.float))
    torch.manual_seed(64439)
    m = Model("base_config")

    # for i in gs:
    #     print(i[1])
    # print(compute_loss(m, gs))
    # out = m(T(game_state), T(guessed, dtype=torch.float))
    # print(out[0].size(), out[1].size(), end = ' ')
    # # print(torch.cat((out[0], out[1]), 1).size())
    # print(out[2].size())
    out = m(T(game_state), T(guessed, dtype=torch.float))
    print(len(out[0][0]))
    





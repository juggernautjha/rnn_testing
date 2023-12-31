import numpy as np
import keras
from keras import backend as K
from keras.layers import Input, Dense, Embedding
from keras.layers import Concatenate, Bidirectional, LSTM, GlobalAveragePooling1D
from keras.models import Model
import yaml

#! Keras implementation of rnn_testing/pytorch/model.py

CONFIGS = "./configs"

class KerasModel():
    """
    *Basically just going to clone the pytorch model.
    """
    def __init__(self, config):
        stream = open(f"{CONFIGS}/{config}.yaml", "r").read()
        self.config = yaml.load(stream, Loader=yaml.Loader)
        # config = dict(self.config)
        d = []
        for i in self.config:
            d.append(list(i.items())[0])
        self.config = dict(d)
        config = self.config
        #! DEFINING LSTM
        # def get_lstm_embedding(maxlen):
        #     input_ = Input(shape=(maxlen, ), name="LSTM_INPUT") 
        #     out = Embedding(maxlen, config['embedding_dim'], mask_zero=True, name="LSTM_EMBEDDING")(input_)
        #     out = Bidirectional(LSTM(config['hidden_dim'], dropout=config['dropout'], return_sequences=True), name="LSTM_Layer_1")(out)
        #     for i in range(1, config['n_lstm']):
        #         out = Bidirectional(LSTM(config['hidden_dim'], dropout=config['dropout'], return_sequences=True), name=f"LSTM_Layer_{i+1}")(out)
        #     #! Averaging out the outputs of various LSTMs.
        #     out = GlobalAveragePooling1D(name="AVERAGE")(out) # TODO: try max pooling.
        #     x = Dense(config['dense_dim_lstm'], activation = config['activation'], name="LSTM")(out)
        #     return Model(input_,x , name = 'game_state_encoding')

        def get_lstm_embedding(maxlen):
            layers = [
                        Embedding(maxlen, config['embedding_dim'], mask_zero=True, name="LSTM_EMBEDDING"),
                        Bidirectional(LSTM(config['hidden_dim'], dropout=config['dropout'], return_sequences=True), name="LSTM_Layer_1")
                    ]
            lstm_units = [
                              Bidirectional(LSTM(config['hidden_dim'], dropout=config['dropout'], return_sequences=True), name=f"LSTM_Layer_{i+1}") for i in range(1, config['n_lstm'])
                        ]
            layers += lstm_units
            layers += [
                GlobalAveragePooling1D(name="AVERAGE"),
                Dense(config['dense_dim_lstm'], activation = config['activation'], name="LSTM")
            ]

            model = keras.Sequential(layers, "game_state_encoding")
            return model
            # out = Embedding(maxlen, config['embedding_dim'], mask_zero=True, name="LSTM_EMBEDDING")(input_)
            # out = Bidirectional(LSTM(config['hidden_dim'], dropout=config['dropout'], return_sequences=True), name="LSTM_Layer_1")(out)
            # for i in range(1, config['n_lstm']):
            #     out = Bidirectional(LSTM(config['hidden_dim'], dropout=config['dropout'], return_sequences=True), name=f"LSTM_Layer_{i+1}")(out)
            # #! Averaging out the outputs of various LSTMs.
            # out = GlobalAveragePooling1D(name="AVERAGE")(out) # TODO: try max pooling.
            # x = Dense(config['dense_dim_lstm'], activation = config['activation'], name="LSTM")(out)
            # return Model(input_,x , name = 'game_state_encoding')
        #! DEFINING GUESSED ENCODING
        def get_guessed_encoding():
            input_ = Input(shape=(26, ), name="GUESSED_INPUT") 
            # out = Embedding(maxlen, config['embedding_dim'], mask_zero=True, name="LSTM_EMBEDDING")(inp)
            # out = Bidirectional(LSTM(config['hidden_dim'], dropout=config['drouput'], return_sequences=True), name="LSTM_Layer_1")(out)
            layers = [
                Dense(config['dense_dim_guessed'], activation = config['activation'], name="GUESSED_DENSE_1"),
            ]
            out = Dense(config['dense_dim_guessed'], activation = config['activation'], name="GUESSED_DENSE_1")(input_)

            for i in range(1, config['n_dense_guessed']):
                out = Dense(config['dense_dim_guessed'], activation = config['activation'], name=f"GUESSED_DENSE_{i+1}")(out)
            #! Averaging out the outputs of various LSTMs.
            return Model(input_,out , name = 'guessed_state_encoding')


        #! DEFINING THE FINAL MODEL
        game_state_encoding = get_lstm_embedding(config['max_len'])
        guessed_state_encoding = get_guessed_encoding()
        true_input = Concatenate()([game_state_encoding.output, guessed_state_encoding.output])
        out = Dense(config['dense_dim_final'], activation = config['activation'], name="FINAL_GUESS")(true_input)
        out = Dense(26, activation='softmax', name="FINAL_PROB_VECTOR")(out)
        self.guessing_model = Model([game_state_encoding.input, guessed_state_encoding.input], out, name="GUESSING_MODEL")
        self.guessing_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=1e-3))


    def __call__(self, game_state_encoding, guessed_state_encoding):
        return self.guessing_model.predict([game_state_encoding, guessed_state_encoding], verbose='false').flatten()

if __name__=='__main__':
    k = KerasModel("base_config")
    k.guessing_model.summary()



"""
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
"""

import numpy as np
from typing import List, Union
import tensorflow as tf
from keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU
from keras.models import Sequential
from keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam, RMSprop
import keras_tuner as kt
import global_var
import random


class NN:
    """
    This class allows to represent the Deep Neural Network that the TradingBot will use for the classification of direction return
    """
    def __init__(
            self,
            layer: str = 'DNN',
            hidden_layers: int = 1,
            activation: str = 'elu',
            seed: int = 0,
            hidden_dim: int = 32,
            epochs: int = 10000,
            learning_rate: float = 0.001,
            rate_dropout: float = 0.3,
            l1: float = 0.0005,
            l2: float = 0.001,
            opt: str = 'RMSprop'
    ):
        self._layer = layer
        self.__hidden_layers = hidden_layers
        self.__activation = activation
        self.__seed = seed
        self.__hidden_dim = hidden_dim
        self._epochs = epochs
        self.__learning_rate = learning_rate
        self.__rate_dropout = rate_dropout
        self.__l1 = l1
        self.__l2 = l2
        self._opt = opt
        self.__nb_features = global_var.nb_features
        self.__optimiz()

    def _instantiate_hp(self, hp: kt.HyperParameters):
        """
        Allows to instantiate the best hyper-parameters value from the tuner
        """
        self._layer = hp.values.get("model_type")
        self.__hidden_layers = hp.values.get("hidden_layers")
        self.__hidden_dim = hp.values.get("hidden_dim")
        self.__activation = hp.values.get("activation")
        self.__learning_rate = hp.values.get("lr")
        self._opt = hp.values.get("optimizer")
        self.__rate_dropout = hp.values.get("rate_dropout")
        self.__l1 = hp.values.get("l1")
        self.__l2 = hp.values.get("l2")
        self.__optimiz()

    def __optimiz(self) -> Union[Adam, RMSprop]:
        if self._opt == 'Adam':
            self._optimizer = Adam(learning_rate=self.__learning_rate)
        elif self._opt == 'RMSprop':
            self._optimizer = RMSprop(learning_rate=self.__learning_rate)
        else:
            raise ValueError('Only Adam and RMSprop optimizer have been implemented')

    def _compute_model(self) -> Sequential:
        """
        Neural network that takes a state and outputs the task asked
        """
        random.seed(self.__seed)
        np.random.seed(self.__seed)
        tf.random.set_seed(self.__seed)
        neural_networks = Sequential()

        if self._layer == 'DNN':
            layer = Dense
            neural_networks.add(
                layer(self.__hidden_dim, input_dim=self.__nb_features, activation=self.__activation,
                activity_regularizer=l1_l2(l1=self.__l1, l2=self.__l2)))
            neural_networks.add(Dropout(self.__rate_dropout, seed=self.__seed))
            for _ in range(self.__hidden_layers):
                neural_networks.add(
                    layer(self.__hidden_dim, activation=self.__activation,
                          activity_regularizer=l1_l2(l1=self.__l1, l2=self.__l2)))
                neural_networks.add(Dropout(self.__rate_dropout, seed=self.__seed))
        else:
            if self._layer == 'SimpleRNN':
                layer = SimpleRNN
            elif self._layer == 'LSTM':
                layer = LSTM
            elif self._layer == 'GRU':
                layer = GRU
            else:
                raise ValueError('The layer should be a SimpleRNN, a LSTM or a GRU')
            neural_networks.add(
                layer(self.__hidden_dim, input_shape=(1, self.__nb_features),
                      return_sequences=True, activity_regularizer=l1_l2(l1=self.__l1, l2=self.__l2)))
            neural_networks.add(Dropout(self.__rate_dropout, seed=self.__seed))
            if self.__hidden_layers >= 2:
                for _ in range(1, self.__hidden_layers):
                    neural_networks.add(
                        layer(self.__hidden_dim,  return_sequences=True, activity_regularizer=l1_l2(l1=self.__l1, l2=self.__l2)))
                    neural_networks.add(Dropout(self.__rate_dropout, seed=self.__seed))
            neural_networks.add(layer(self.__hidden_dim))

        neural_networks.add(Dense(3, activation='softmax'))
        neural_networks.compile(optimizer=self._optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        #neural_networks.summary()
        return neural_networks
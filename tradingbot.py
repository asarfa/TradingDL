import numpy as np
import pandas as pd
from typing import List
from neural_network import NN
from finance import MarketEnv
import keras_tuner as kt
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder

class TradingBot(NN):
    """
    This class allows to represent the trading bot agent based on the Neural Network class
    """
    def __init__(
            self,
            learn_env: MarketEnv = None,
            valid_env: MarketEnv = None,
            hypermodel: kt.HyperModel = None,
            hp_opt=None,
            verbose=True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.__learn_env = learn_env
        self.valid_env = valid_env
        self.__hp_opt = hp_opt
        self.__verbose = verbose
        self.__set_model(hypermodel)

    def __set_model(self, hypermodel: kt.HyperModel = None):
        """
        Set the model depending on hyperopt or not
        """
        if not self.__hp_opt:
            self.model = self._compute_model()
        else:
            self._instantiate_hp(self.__hp_opt)
            if not hypermodel is None:
                self.model = hypermodel
            else:
                if self.__verbose: print(f"Model's hyper-parameters: {self.__hp_opt.values}")
                self.model = self._compute_model()

    @property
    def callbacks(self) -> List[EarlyStopping]:
        return [EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)]

    @staticmethod
    def __transform_targetRNN(target: pd.Series) -> np.ndarray:
        return target.values.reshape(-1, 1)

    @staticmethod
    def __transform_stateRNN(state: pd.DataFrame) -> np.ndarray:
        return state.values.reshape(state.shape[0], 1, state.shape[1])

    def compute_fit(self, state: pd.DataFrame, target: pd.Series, cw: bool = True, ohe_done: bool = False):
        """
        Fitting the model
        """
        if self._layer != 'DNN':
            state = self.__transform_stateRNN(state)
        if not ohe_done:
            target = OneHotEncoder(sparse=False, dtype=int).fit_transform(self.__transform_targetRNN(target))
        if self.__learn_env.cw and cw: #Classification
            self.model.fit(state, target, epochs=self._epochs, shuffle=False, verbose=False,
                           class_weight=self.__learn_env.cw, callbacks=self.callbacks)
        else:
            self.model.fit(state, target, epochs=self._epochs, shuffle=False, verbose=False, callbacks=self.callbacks)

    def compute_prediction(self, state: pd.DataFrame) -> np.ndarray:
        """
        Predicting the returns direction classification
        Softmax function will output a probability of class membership for each class label
        Therefore we will select the one with the biggest probability
        """
        if self._layer != 'DNN':
            state = self.__transform_stateRNN(state)
        prediction = self.model.predict(state, verbose=False, batch_size=None, callbacks=self.callbacks)
        return np.argmax(prediction, axis=1)

    def compute_score(self, y, prediction) -> float:
        return sum(prediction == y.values)/len(prediction)

    def __fit_model(self):
        """
        Method to fit the performance of the trading agent.
        """
        self.compute_fit(self.__learn_env.X, self.__learn_env.y)
        prediction = self.compute_prediction(self.__learn_env.X)
        print(f'************************ Accuracy in sample: {self.compute_score(self.__learn_env.y, prediction)} ************************')

    @staticmethod
    def reset_env(env: MarketEnv):
        """
        Method to erase the columns created by precedent model strategy from backtest
        """
        try:
            env.data.drop(columns=['position', 'strategy', 'strategy_tc', 'cum_returns', 'cum_strategy',
                                              'cum_strategy_tc'], inplace=True)
        except KeyError:
            pass

    def __valid_model(self):
        """
        Method to validate the performance of the trading agent.
        """
        self.reset_env(self.valid_env)
        prediction = self.compute_prediction(self.valid_env.X)
        print(f'************************ Accuracy validation sample: {self.compute_score(self.valid_env.y, prediction)} ************************')

    def learn(self, valid: bool = True):
        self.__fit_model()
        if valid: self.__valid_model()

    def compute_position(self, prediction: np.array) -> np.ndarray:
        prediction = prediction
        position = np.zeros(len(prediction))
        position[prediction == 0] = -1
        position[prediction == 2] = 1
        return position
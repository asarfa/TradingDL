import pandas as pd
from neural_network import NN
import keras_tuner as kt
from tradingbot import TradingBot
from vect_backtesting import backtest
import json


class MyHyperModel(kt.HyperModel):
    """
    This class aims at finding the hyper-parameters which lead to the neural network with the biggest performance on the validation set
    """

    def build(self, hp):
        """
        Define the space search of hyper-parameters and the same neural network model as in the neural_network class
        """
        seed = 0
        layer = hp.Choice("model_type", ["DNN", "LSTM", "SimpleRNN", "GRU"])
        hidden_layers = hp.Int("hidden_layers", 1, 3)
        hidden_dim = hp.Int("hidden_dim", min_value=32, max_value=256, step=32)
        activation = hp.Choice("activation", ["relu", "elu", "selu"])
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        opt = hp.Choice("optimizer", ["Adam", "RMSprop"])
        rate_dropout = hp.Float("rate_dropout", min_value=0.1, max_value=0.4, sampling="log")
        l1 = hp.Float("l1", min_value=0.0005, max_value=0.001, sampling="log")
        l2 = hp.Float("l2", min_value=0.001, max_value=0.01, sampling="log")
        neural_net = NN(layer=layer, hidden_layers=hidden_layers, activation=activation, seed=seed,
           hidden_dim=hidden_dim, learning_rate=learning_rate, rate_dropout=rate_dropout, l1=l1, l2=l2, opt=opt)
        return neural_net._compute_model()

    def fit(self, hp, model, learn_env, valid_env, **kwargs):
        """
        Evaluation of the model and return the objective value to minimize
        """
        agent = TradingBot(learn_env, valid_env, hp_opt=hp, hypermodel=model)
        agent.learn(valid=False)
        env_val = agent.valid_env
        VL_strat, acc = backtest(env_val, agent)
        return {'VL_strat_tc': VL_strat, 'accuracy': acc}


def analyse_trials_opt(max_trials: int, directory: str, project_name: str):
    """
    Analysing the tuning, finding the hyper-parameters leading to the maximal performance on validation set
    """
    list_trials = []
    len_max = len(str(max_trials))
    for num_trial in range(max_trials):
        len_diff = len_max - len(str(num_trial))
        if len_diff != 0:
            str_num_trial = '0' * len_diff + str(num_trial)
        else:
            str_num_trial = str(num_trial)
        try:
            with open(directory + '/' + project_name + '/trial_' + str_num_trial + '/trial.json') as f:
                trial = json.load(f)
            found = True
        except FileNotFoundError:
            found = False
        if found:
            trial_hp = pd.DataFrame(trial['hyperparameters']['values'].values(), index=trial['hyperparameters']['values'].keys()).T
            trial_hp['accuracy'] = trial['metrics']['metrics']['accuracy']['observations'][0]['value'][0]
            trial_hp['performance'] = trial['score'] / 100 - 1
            trial_hp['status'] = trial['status']
            list_trials.append(trial_hp)
    trials = pd.concat(list_trials, ignore_index=True)
    trials = trials.sort_values(by='performance', ascending=False)
    print(100 * '-')
    print('Trials configuration maximizing the performance of the bot strategy with transaction cost on the validation sample: ')
    print(100 * '-')
    print(trials)
    return trials

class HP:
    """
    This class allows to create a proxy of the hp class from keras tuner
    The values are the hyper-parameters of a given trial
    """

    def __init__(
            self,
            best_trials: pd.DataFrame = None,
            index: int = None
    ):
        self.__best_trials = best_trials
        self.__index = index

    @property
    def values(self):
        return self.__best_trials.iloc[self.__index][:-len(['accuracy', 'performance', 'status'])].to_dict()

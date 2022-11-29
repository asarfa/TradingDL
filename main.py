import pandas as pd
import keras_tuner as kt
from statsmodels.tsa.stattools import pacf
from finance import MarketEnv
from tradingbot import TradingBot
from vect_backtesting import backtest
from hypertuning import MyHyperModel, analyse_trials_opt, HP
from ensemble_learning import EnsembleLearning
from EnumClasses import EnumPair
from QuotesReaderOneCCY import QuotesReaderOneCCY
from datetime import datetime, timedelta
import global_var
import os
import sys

"""
The aim of this project is to create a Neural Network strategy through a trading bot which classify the direction of 
returns in order to take action (long, neutral, short) by interacting with a financial market environment and then backtesting
this strategy
"""

def _transform_csv(currencies: list = None):
    """
    Intraday forex data of multiple currencies convert to a DataFrame
    """
    file_name = "data.part05/live-fm-log-11May-07-20-17-022.csv"
    list_data=[]
    for currency in currencies:
        if currency == 'GBPUSD':
            currency_pair = EnumPair.GBPUSD
        elif currency == 'USDCHF':
            currency_pair = EnumPair.GBPAUD
        elif currency == 'USDJPY':
            currency_pair = EnumPair.USDJPY
        else:
            raise Exception(f'{currency} has not been implemented as a currency')
        liste_quote = QuotesReaderOneCCY(file_name, currency_pair).read_to_end()
        data = pd.DataFrame(list(
            map(lambda quote: [datetime.fromtimestamp(0) + timedelta(microseconds=quote.get_ecn_timestamp()),
                               quote.get_price()], liste_quote)), columns=['Date', 'Close'])
        data.drop_duplicates(subset=['Date'], keep='last', inplace=True)
        data.set_index('Date', inplace=True)
        data.ffill(inplace=True)
        data.columns = pd.MultiIndex.from_product([['GBPUSD'], data.columns])
        list_data.append(data)
    data = pd.concat(list_data, axis=1, join='outer')
    data.to_hdf(f'data.used/{"_".join(currencies)}.h5', 'df')
    return data

def _load_data(currencies: list = None):
    """
    Load data under a Dataframe
    """
    try:
        data = pd.read_hdf(f'data.used/{"_".join(currencies)}.h5', 'df')
    except FileNotFoundError:
        data = _transform_csv(currencies)
    return data

def load_data(currencies: list = None):
    """
    For each ticker in a Dataframe load the asset features in a dict
    """
    data = _load_data(currencies)
    print(f'Number of ticks per assets: {len(data)}')
    dict_ticks = dict(map(lambda tick: (tick, data[tick].dropna()), data.columns.levels[0]))
    return dict_ticks

def find_lags_env(data: dict, tick: str, start: int, end: int):
    """
    Find for the training sample the last return's index which has a significant autocorrelation with the first return
    """
    max_ = int(len(data[tick].iloc[start: end])*0.1) if int(len(data[tick].iloc[start: end])*0.1) <= 10 else 10
    pacf_ = pacf(data[tick]['Close'].pct_change().dropna().iloc[start: end].values.squeeze(), nlags=max_)
    last_significant = [i for i, x in enumerate(abs(pacf_) >= 0.05) if x][-1]
    if last_significant==0: last_significant=1
    return last_significant

if __name__ == '__main__':

    """
    Instantiation of the data (intraday price of currencies)
    """
    data = load_data(currencies=['GBPUSD'])

    """
    Tickers data contained in the dict
    """
    tickers = list(data.keys())

    """
    Instantiation of different samples
    Training data will be 70% of the original data
    Validation and testing data will be each 15%
    """
    train_sample, val_sample, test_sample = 0.70, 0.15, 0.15
    start_train = 0
    end_train = int(len(data[tickers[0]])*train_sample)
    start_val = end_train + 1
    end_val = end_train + int(len(data[tickers[0]])*val_sample)
    start_test = end_val + 1
    end_test = end_val + int(len(data[tickers[0]])*test_sample)

    """
    Type of tuners to use in the hyper-optimization
    """
    type_tuner = 'random' #random or bayesian

    """ 
    For each tickers in the initial dataframe, computing the next steps
    """
    for tick in tickers:

        global_var.id = type_tuner + '_tuner/' + tick + '_' + str(len(data[tick]))
        output_dir = f'output_console/{global_var.id}.txt'

        if not os.path.isfile(output_dir):
            sys.stdout = open(output_dir, 'w') #The output of the consol is print in a file in order to save results

        print(f'Start of the execution: {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}')
        print(100 * '-')
        print(tick)
        print(100 * '-')

        """
        Finding the length per environment
        """
        nb_ticks_env = find_lags_env(data, tick, start_train, end_train)

        """
        Instantiation of the finance environment for training, validation, testing
        """
        learn_env = MarketEnv(data, tick, nb_ticks_env, start_train, end_train)
        valid_env = MarketEnv(data, tick, nb_ticks_env, start_val, end_val, scaler_train=learn_env.scaler_X, features=learn_env.features)
        test_env = MarketEnv(data, tick, nb_ticks_env, start_test, end_test, scaler_train=learn_env.scaler_X, features=learn_env.features)

        """
        Features used in the model
        """
        features = learn_env.features
        print(f'There are {len(features)} features: {features}')
        global_var.nb_features = len(features)

        """
        Computation of the Trading bot without tuning hyperparameters
        _Fitting
        _Validation
        
        agent = TradingBot(learn_env, valid_env)
        agent.learn()
        """

        """
        Tuning of the hyper-parameters in order to find the neural network which lead to the biggest performance on 
        the validation set
        As the tuning is long we already tuned the hyperparameters and saved the best trial
        """
        directory = 'hyperopt'
        project_name = global_var.id
        max_trials = 100
        if not os.path.isdir(f'{directory}/{project_name}'):
            if type_tuner == 'random':
                tuner = kt.RandomSearch(
                    hypermodel=MyHyperModel(),
                    objective=kt.Objective("VL_strat_tc", "max"),
                    max_trials=max_trials,
                    directory=directory,
                    overwrite=True,
                    project_name=project_name
                )
            elif type_tuner == 'bayesian':
                tuner = kt.BayesianOptimization(
                    hypermodel=MyHyperModel(),
                    objective=kt.Objective("VL_strat_tc", "max"),
                    max_trials=max_trials,
                    directory=directory,
                    overwrite=True,
                    project_name=project_name
                )
            print(tuner.search_space_summary())
            tuner.search(learn_env=learn_env, valid_env=valid_env)

        """
        Loading the best models of hyperopt from the json save in the directory
        """
        best_trials = analyse_trials_opt(max_trials, directory, project_name)

        """
        Loading the best set of hyper-parameters on the validation set
        """
        best_hp = HP(best_trials, 0)

        """
        Training of the trading bot with the best set of hp
        """
        agent = TradingBot(learn_env, valid_env, hp_opt=best_hp)
        agent.learn(valid=False)

        """
        Testing of the trading bot and plotting its performance from the vectorized-backtester
        """
        VL_strat, acc = backtest(test_env, agent, graph=True, type_env='test')

        """
        Testing of the trading bot and plotting its gross performance with EnsembleLearning: multiple models
        """
        ens_model = EnsembleLearning(learn_env, valid_env, test_env, nb_ticks_env, features, best_trials, nb_models=3)
        ens_model.compute()
        """
        Testing of the trading bot and plotting its gross performance with EnsembleLearning: multiple seeds
        """
        ens_seeds = EnsembleLearning(learn_env, valid_env, test_env, nb_ticks_env, features, best_trials, nb_seeds=5)
        ens_seeds.compute()
        """
        Testing of the trading bot and plotting its gross performance with EnsembleLearning: multiple X
        """
        ens_X = EnsembleLearning(learn_env, valid_env, test_env, nb_ticks_env, features, best_trials, pct_drop_X=0.3)
        ens_X.compute()
        """
        Testing of the trading bot and plotting its gross performance with EnsembleLearning: all methods
        """
        ens_ens = EnsembleLearning(learn_env, valid_env, test_env, nb_ticks_env, features, best_trials, nb_models=3, nb_seeds=5, pct_drop_X=0.3)
        ens_ens.compute(ens_model.ensemble_positions, ens_seeds.ensemble_positions, ens_X.ensemble_positions)

        print(f' End of the execution: {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}')
        sys.stdout.close()
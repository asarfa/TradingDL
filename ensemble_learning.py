import pandas as pd
import numpy as np
import random
from finance import MarketEnv
from tradingbot import TradingBot
from vect_backtesting import backtest
from hypertuning import HP


class EnsembleLearning:
    """The aim of this class is to compute some ensemble learning by compiling the predictions of the best models"""
    def __init__(
            self,
            learn_env: MarketEnv = None,
            valid_env: MarketEnv = None,
            test_env: MarketEnv = None,
            nb_ticks_env: int = None,
            features: list = None,
            best_trials: pd.DataFrame = None,
            nb_models: int = None,
            nb_seeds: int = None,
            pct_drop_X: float = None
    ):
        self.__learn_env = learn_env
        self.__valid_env = valid_env
        self.test_env = test_env
        self.__nb_ticks_env = nb_ticks_env
        self.__features = features
        self.__best_trials = best_trials
        self.__nb_models = nb_models
        self.__nb_seeds = nb_seeds
        self.__pct_drop_X = pct_drop_X

    @property
    def type(self):
        if (self.__nb_models and self.__nb_seeds) or (self.__nb_models and self.__pct_drop_X) or \
                (self.__nb_seeds and self.__pct_drop_X) or (self.__nb_models and self.__nb_seeds and self.__pct_drop_X):
            return 'ensemble_of_ensemble'
        else:
            return 'ensemble'

    @property
    def title(self):
        if self.type == 'ensemble':
            if self.__nb_models:
                return 'Multiple Models'
            if self.__nb_seeds:
                return 'Multiple Seeds'
            if self.__pct_drop_X:
                return 'Multiple Training'
        else:
            return 'Multiple Methods'

    @property
    def best_hps(self):
        nb_models = self.__nb_models if self.__nb_models else 1
        return list(map(lambda index: HP(self.__best_trials, index), range(nb_models)))

    def __steps(self, ensemble_positions, agent):
        agent.learn(valid=False)
        predictions = agent.compute_prediction(self.test_env.X)
        positions = agent.compute_position(predictions)
        ensemble_positions.append(positions.flatten())
        return ensemble_positions

    def __multiple_models(self, ensemble_positions: list):
        print(100 * '-')
        print(f'Ensemble Learning will be computed with the top {self.__nb_models} models')
        print(100 * '-')
        for hp in self.best_hps:
            agent = TradingBot(self.__learn_env, self.__valid_env, hp_opt=hp)
            ensemble_positions = self.__steps(ensemble_positions, agent)
        return agent, ensemble_positions

    def __multiple_seeds(self, ensemble_positions: list):
        print(100 * '-')
        print(f'Ensemble Learning will be computed with the best model under {self.__nb_seeds} different random seeds')
        print(100 * '-')
        seeds = random.sample(range(100), self.__nb_seeds)
        print(f'Seeds used: {seeds}')
        for seed in seeds:
            agent = TradingBot(self.__learn_env, self.__valid_env, hp_opt=self.best_hps[0], seed=seed, verbose=False)
            ensemble_positions = self.__steps(ensemble_positions, agent)
        return agent, ensemble_positions

    def __varying_models(self, ensemble_positions: list):
        if self.__nb_seeds is None:
            agent, ensemble_positions = self.__multiple_models(ensemble_positions)
        else:
            agent, ensemble_positions = self.__multiple_seeds(ensemble_positions)
        return agent, ensemble_positions

    def __varying_training(self, ensemble_positions: list):
        nb_subset = 5
        print(100 * '-')
        print(f'Ensemble Learning will be computed with the best model dropping randomly {self.__pct_drop_X * 100}% of features, {nb_subset} times fitted')
        print(100 * '-')
        X_learn_env = self.__learn_env.X.copy()
        X_test_env = self.test_env.X.copy()
        for n in range(nb_subset):
            random.seed(n)
            idx_features_0 = random.sample(range(len(self.__features)), int(len(self.__features)*self.__pct_drop_X))
            features_not_0 = [j for i, j in enumerate(self.__features) if i not in idx_features_0]
            if self.__pct_drop_X >= 0.5:
                print(f'Features not set to 0: {features_not_0}')
            else:
                print(f'Features set to 0: {np.array(self.__features)[idx_features_0]}')
            self.__learn_env.X.iloc[:, idx_features_0] = 0
            self.test_env.X.iloc[:, idx_features_0] = 0
            agent = TradingBot(self.__learn_env, self.__valid_env, hp_opt=self.best_hps[0], verbose=False)
            ensemble_positions = self.__steps(ensemble_positions, agent)
            self.__learn_env.X = X_learn_env.copy()
            self.test_env.X = X_test_env.copy()
        return agent, ensemble_positions

    def __to_df(self, ensemble_pos: list):
        return pd.DataFrame(ensemble_pos, index=range(len(ensemble_pos))).T

    def __ensemble(self):
        ensemble_positions = []
        if self.__pct_drop_X is None:
            agent, ensemble_positions = self.__varying_models(ensemble_positions)
        else:
            agent, ensemble_positions = self.__varying_training(ensemble_positions)
        print('For each bar of the test environnement, the position taken is the one with the biggest occurrence from all networks of the ensemble method used above')
        self.ensemble_positions = self.__to_df(ensemble_positions)
        return agent, self.ensemble_positions

    def __ensemble_of_ensemble(self, ens_pos_models: pd.DataFrame = None, ens_pos_seeds: pd.DataFrame = None, ens_pos_X: pd.DataFrame = None):
        print(100 * '-')
        print(100 * '-')
        print(f'Ensemble Learning will be computed as an ensemble of the upcoming multiple methods: ')
        print(100 * '-')
        ensemble_positions_models, ensemble_positions_seeds, ensemble_positions_trainings = [], [], []
        if self.__nb_models and not isinstance(ens_pos_models, pd.DataFrame):
            agent, ensemble_positions_models = self.__multiple_models(ensemble_positions_models)
        elif self.__nb_seeds and not isinstance(ens_pos_seeds, pd.DataFrame):
            agent, ensemble_positions_seeds = self.__multiple_seeds(ensemble_positions_seeds)
        elif self.__pct_drop_X and not isinstance(ens_pos_X, pd.DataFrame):
            agent, ensemble_positions_trainings = self.__varying_training(ensemble_positions_trainings)
        else:
            agent = TradingBot(self.__learn_env, self.__valid_env, hp_opt=self.best_hps[0], verbose=False)
        ensemble_positions_models = ens_pos_models if isinstance(ens_pos_models, pd.DataFrame) else self.__to_df(ensemble_positions_models)
        ensemble_positions_seeds = ens_pos_seeds if isinstance(ens_pos_seeds, pd.DataFrame) else self.__to_df(ensemble_positions_seeds)
        ensemble_positions_trainings = ens_pos_X if isinstance(ens_pos_X, pd.DataFrame) else self.__to_df(ensemble_positions_trainings)
        print('For each bar of the test environnement, the position taken is the one with the biggest occurrence from all networks of the mutliple ensemble methods used above')
        ensemble_of_ensemble_positions = pd.concat([ensemble_positions_models, ensemble_positions_seeds, ensemble_positions_trainings], axis=1)
        return agent, ensemble_of_ensemble_positions

    def compute(self, ens_pos_models: pd.DataFrame = None, ens_pos_seeds: pd.DataFrame = None, ens_pos_X: pd.DataFrame = None):
        if self.type == 'ensemble':
            agent, ensemble_positions = self.__ensemble()
        else:
            agent, ensemble_positions = self.__ensemble_of_ensemble(ens_pos_models, ens_pos_seeds, ens_pos_X)
        positions = list(map(lambda index: ensemble_positions.iloc[index].value_counts().index[0], range(len(ensemble_positions))))
        self.test_env.data['position'] = positions
        backtest(self.test_env, agent, graph=True, type_env='test', type_bt=self.title, position_done=True)
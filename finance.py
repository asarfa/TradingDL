import pandas as pd
import numpy as np
import talib
from talib.abstract import Function
"""
Used the talib package which perform technical analysis of financial market data and help identifying different patterns
that stocks follow
https://blog.quantinsti.com/install-ta-lib-python/
https://www.lfd.uci.edu/~gohlke/pythonlibs/#_ta-lib
TA_Lib‑0.4.24‑cp37‑cp37m‑win_amd64.whl
"""
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pickle
import global_var
pd.options.mode.chained_assignment = None  # Ignore Setting With Copy Warning

class MarketEnv:
    """
    This class allows to represent the financial market environment
    Prepares the features and scaled them
    """
    def __init__(
            self,
            dict_ticks: dict = None,
            symbol: str = None,
            lags_env: int = None,
            start: int = None,
            end: int = None,
            lower_quantile: float = 0.15,
            upper_quantile: float = 0.85,
            leverage: float = 1,
            type_scaling: str = 'normalized',
            scaler_train=None,
            features: list = None,
            source: str = None,
            one_lag_max: bool = False

    ):
        self.__dict_ticks = dict_ticks
        self.symbol = symbol
        self.lags_env = lags_env
        self.__start = start
        self.__end = end
        self.__lower_quantile = lower_quantile
        self.__upper_quantile = upper_quantile
        self.leverage = leverage
        self.features = features
        self.__source = source
        self.__one_lag_max = one_lag_max
        self.__set_args(type_scaling, scaler_train)

    def __create_class(self, df):
        """
        Create a multi class label corresponding to a short, neutral or long position based on the lower and
        upper percentile of returns
        For example if lower_quantile = 0.25 and upper quantile = 0.75:
        25% of the direction will be short, represented by 0
        50% of the direction will be neutral, no need to take a position for very small profit, represented by 1
        25% of the direction will be long, represented by 2
        As the class are imbalanced a method to compute cw will be instantiated
        """
        ret = df['returns']
        lower = ret.quantile(self.__lower_quantile)
        upper = ret.quantile(self.__upper_quantile)
        short_index = ret[ret < lower].index
        long_index = ret[ret > upper].index
        df['direction'] = 1
        df['direction'][short_index] = 0
        df['direction'][long_index] = 2
        return df

    def __manual_features(self, df: pd.DataFrame, max_rolling: int) -> pd.DataFrame:
        df['Open'] = 0
        df['Open'].iloc[max_rolling:] = df['Close'].iloc[:-max_rolling]
        df['Low'] = df['Close'].rolling(max_rolling).min()
        df['High'] = df['Close'].rolling(max_rolling).max()
        df['returns'] = df['Close'] / df['Close'].shift() - 1
        df = self.__create_class(df)
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift())
        df['volatility'] = df['log_returns'].rolling(14).std()
        df['distance'] = df['Close'] - df['Close'].rolling(max_rolling).mean()
        df['min'] = df['returns'].rolling(max_rolling).min()
        df['max'] = df['returns'].rolling(max_rolling).max()
        df['mami'] = df['max'] - df['min']
        df['mac'] = abs(df['max'] - df['returns'].shift())
        df['mic'] = abs(df['min'] - df['returns'].shift())
        df['atr'] = np.maximum(df['mami'], df['mac'])
        df['atr'] = np.maximum(df['atr'], df['mic'])
        df['atr%'] = df['atr']/df['Close']
        return df

    def __talib_features(self, df: pd.DataFrame, max_rolling: int) -> pd.DataFrame:
        overlap_studies = {'sma': max_rolling, 'dema': max_rolling, 'ema': max_rolling, 'ht_trendline': max_rolling,
                           'kama': max_rolling, 'ma': max_rolling/3, 'ma': 2*max_rolling/3, 'ma': max_rolling,
                           'midpoint': 14, 't3': 5, 'tema': max_rolling, 'trima': max_rolling, 'wma': max_rolling}
        momentum_indic = {'adx': 14, 'adxr': 14, 'aroonosc': 14, 'cci': 14, 'cmo': 14, 'dx': 14,
                          'plus_di': 14, 'plus_dm': 14, 'rsi': 14, 'willr': 14}
        talib_field = {}
        for d in (overlap_studies, momentum_indic):
            talib_field.update(d)
        talib_pattern_regonition = talib.get_function_groups()['Pattern Recognition']
        inputs = {
            'close': df['Close'],
            'high': df['High'],
            'low': df['Low'],
            'open': df['Open']
        }
        for field, timeperiod in talib_field.items():
            df[f'{field}_{timeperiod}'] = Function(field)(inputs, timeperiod=timeperiod)
        for field in talib_pattern_regonition:
            df[f'{field}'] = Function(field)(inputs)
        return df

    def __prepare_features(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepares the feature columns to feed the NN, manual making and talib making
        Features are prepared considering a max window of rolling (default=60)
        Indeed
        As the ticks are intraday:
        Close is considered as the actual tick (in micro second)
        Open is considered as the - max_rolling Close ticks before
        High is considered as the max Close of the max_rolling data
        Low is considered as the min Close of the max_rolling data
        """
        max_rolling = 30
        df = self.__dict_ticks[self.symbol].iloc[self.__start: self.__end].copy()
        df = self.__manual_features(df, max_rolling)
        df = self.__talib_features(df, max_rolling)
        if self.features is None:
            """
            Delete fields which are useless i.e only 0 has values
            """
            df.drop(columns=list(df.iloc[:, ((df==0).sum()==len(df)).values].columns), inplace=True)
        characteristics = list(df.columns)[len(['Open', 'High', 'Low', 'Close']):]
        return df.dropna(), characteristics

    def __init_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        As we want to forecast y (returns) with known values at the date of the forecasting, we have to make shift operations
        Shift(1) is done for each feature
        Shift(1) until Shift(lags_env) is done only for returns and direction features
        """
        df, characteristics = self.__prepare_features()
        true_features = []
        lags_env = 1 if self.__one_lag_max else self.lags_env
        for feature in characteristics:
            for lag in range(1, lags_env+1):
                feature_str = f'{feature}_lag_{lag}'
                if lag > 1 and feature != "direction":
                    pass
                else:
                    df[feature_str] = df[feature].shift(lag)
                    true_features.append(feature_str)
            if feature not in ["returns", "direction"]: df.drop(feature, axis=1, inplace=True)
        return df.dropna(), true_features

    @property
    def actions(self) -> List[str]:
        return ['short', 'neutral', 'long']

    def __compute_cw(self):
        """
        In order to avoid imbalanced label class; i.e more negative returns than positive or vice-versa
        """
        c0, c1, c2 = np.bincount(self.__df['direction'])
        w0 = (1 / c0) * (len(self.__df)) / len(self.actions)
        w1 = (1 / c1) * (len(self.__df)) / len(self.actions)
        w2 = (1 / c2) * (len(self.__df)) / len(self.actions)
        return {0: w0, 1: w1, 2: w2}

    def __init_scaler(self, type: str, scaler_train):
        """
        Type of scaler wanted by the user
        For the validation and testing, the training scaler will be provided
        """
        if scaler_train is None:
            if type == 'normalized':
                return MinMaxScaler()
            elif type == 'standardized':
                return StandardScaler()
            else:
                raise ValueError('type of scaling must be normalized or standardized')
        else:
            return scaler_train

    def __features_from_model(self, model, X: pd.DataFrame, y: pd.Series):
        fs = SelectFromModel(model)
        fs.fit(X, y)
        selected_feat = list(X.columns[(fs.get_support())])
        return selected_feat

    def __features_importance(self, X: pd.DataFrame):
        """
        Method which allows to select the relevant features through ensemble ML classification algorithm RandomForest
        """
        id_ = global_var.id.split('/')[1]

        try:
            with open(f'features_selection/{id_}.pkl', 'rb') as f:
                selected_feat = pickle.load(f)
        except FileNotFoundError:
            list_features = []
            model = RandomForestClassifier(n_estimators=1000, class_weight=self.cw, n_jobs=-1, bootstrap=False)
            list_features.extend(self.__features_from_model(model, X, self.__df['direction']))
            #specifying the type of features in order to get each kind, without it there is no pattern recognition features in the features selected
            indicators = [col for col in X.columns if col[:3] != 'CDL']
            pattern_recognition = [col for col in X.columns if col[:3] == 'CDL']
            list_features.extend(self.__features_from_model(model, X[indicators], self.__df['direction']))
            list_features.extend(self.__features_from_model(model, X[pattern_recognition], self.__df['direction']))

            selected_feat = list(set(list_features))
            with open(f'features_selection/{id_}.pkl', 'wb') as f:
                pickle.dump(selected_feat, f)

        removed_feat = [feat for feat in X.columns if feat not in selected_feat]
        return selected_feat, removed_feat

    def __compute_data(self, scaler_train):
        """
        Return the all data, X scaled (features scaled) and y
        """
        self.__df, features = self.__init_data()
        self.cw = self.__compute_cw()
        df = self.__df.copy()
        if scaler_train is None: #for training sample
            self.features, features_to_del = self.__features_importance(df[features])
            df[self.features] = self.scaler_X.fit_transform(df[self.features])
        else: #for validation and testing sample
            df[self.features] = self.scaler_X.transform(df[self.features])
        y = df['direction']
        return df, df[self.features], y

    def __set_args(self, type_scaling, scaler_train):
        self.scaler_X = self.__init_scaler(type_scaling, scaler_train)
        self.data, self.X, self.y = self.__compute_data(scaler_train)
        self.__n_features = len(self.features)
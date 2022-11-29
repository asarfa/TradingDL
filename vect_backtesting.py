import numpy as np
import pandas as pd
from finance import MarketEnv
from tradingbot import TradingBot
from pylab import plt, mpl
import os
import global_var
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 1000
mpl.rcParams['font.family'] = 'serif'

"""
Function that allows to backtest the trading strategies based on a The TradingBot
Having a model with high accuracy of prediction is important but is generally not enough to generate alpha
Indeed it is important for the TradingBot to predict the large market movements correctly and not just
the majority of the market movements.
Vectorized backtesting is an easy and fast way of figuring out the potential of the TradingBot' performance
"""


def backtest(env: MarketEnv = None, agent: TradingBot = None, tc: float = None, spread: float = 0.00012,
             amount: float = 100, graph: bool = False, type_env: str = 'valid',
             type_bt: str = 'Best Model', position_done: bool = False, verbose: bool = True):

    if position_done:
        prediction = env.data['position'].copy()
        prediction[env.data['position'] == -1] = 0
        prediction[env.data['position'] == 0] = 1
        prediction[env.data['position'] == 1] = 2
        acc = agent.compute_score(env.data['direction'], prediction)
    else:
        agent.reset_env(env)
        prediction = agent.compute_prediction(env.X)
        acc = agent.compute_score(env.data['direction'], prediction)
        env.data['position'] = agent.compute_position(prediction)

    if verbose:
        if type_env == 'test':
            print(f"************************ Accuracy out-of-sample: {acc} ************************")
        else:
            print(f"************************ Accuracy validation sample: {acc} ************************")

    def ptf_base(start_sum, rets):
        for r in rets:
            v = start_sum * (1 + r)
            yield v
            start_sum = v

    env.data['strategy'] = env.data['position'] * env.data['returns'] * env.leverage #Calculates the strategy returns given the position values
    # determine when a trade takes place
    trades = env.data['position'].diff().fillna(1) != 0
    # instantiate strategy with transaction cost
    env.data['strategy_tc'] = env.data['strategy']
    if tc is None:
        #spread = 0.00012 --> bid-ask spread on day trader level
        tc = spread/env.data['Close'].mean()
    # subtract transaction costs from return when trade takes place
    env.data['strategy_tc'][trades] -= tc
    #compute the VL base 100 of the passive returns, strategy and strategy with transaction cost
    env.data['cum_returns'] = pd.Series(list(ptf_base(amount, env.data['returns'])), index=env.data.index)
    env.data['cum_strategy'] = pd.Series(list(ptf_base(amount, env.data['strategy'])), index=env.data.index)
    env.data['cum_strategy_tc'] = pd.Series(list(ptf_base(amount, env.data['strategy_tc'])), index=env.data.index)
    VL_strat = env.data['cum_strategy_tc'].iloc[-1]
    aperf = VL_strat / amount - 1
    operf = aperf - (env.data['cum_returns'].iloc[-1] / amount - 1)
    if verbose:
        print(f'The number of trades is {sum(trades)}, there is a total of {len(env.data)} ticks')
        print('The absolute performance of the strategy with tc is {:.1%}'.format(aperf))
        print('The outperformance of the strategy with tc is {:.1%}'.format(operf))
        print(100 * '*')
    if graph:
        columns = ['cum_returns', 'cum_strategy_tc']
        idx_freq = env.data.index.to_series().diff().value_counts().index[0]
        idx_ant = env.data.index[0] - idx_freq
        VL = pd.DataFrame([amount]*len(columns), columns=[idx_ant], index=columns).T.append(env.data[columns])
        title = '%s | TC = %.5f' % (f'{env.symbol}'
                                    f': {type_bt} VL TradingBot vs Passive', tc)
        VL.plot(title=title, figsize=(10, 6))
        plt.show(block=False)
        directory = f'graphs/{global_var.id}'
        if not os.path.exists(directory):
            os.mkdir(directory)
        plt.savefig(f'{directory}/{type_env}_env_{type_bt}_performance.png')
    return VL_strat, acc
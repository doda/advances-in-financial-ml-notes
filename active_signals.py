'''
Code from Chapter 10 of Advances in Financial Machine Learning
'''

import pandas as pd
from multiprocess import mpPandasObj

def avgActiveSignals(signals, numThreads):
    # compute the average signal among those active
    # 1) time points where signals change (either one starts or one ends)
    tPnts = set(signals['t1'].dropna().values)
    tPnts = tPnts.union(signals.index.values)
    tPnts = list(tPnts)
    tPnts.sort()
    out = mpPandasObj(mpAvgActiveSignals, ('molecule', tPnts), numThreads, signals=signals)
    return out

def mpAvgActiveSignals(signals, molecule):
    '''
    At time loc, average signal among those still active.
    Signal is active if:
      a) issued before or at loc AND
      b) loc before signal's endtime, or endtime is still unknown (NaT).
    '''
    out = pd.Series()
    for loc in molecule:
        df0 = (signals.index.values <= loc) & ((loc < signals['t1']) | pd.isnull(signals['t1']))
        act = signals[df0].index
        if len(act) > 0:
            out[loc] = signals.loc[act, 'signal'].mean()
        else:
            out[loc] = 0
    return out
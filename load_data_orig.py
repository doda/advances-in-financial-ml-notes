import pandas as pd
from path import Path
DATA_DIR = Path('../data/')
BROKEN = {
    '@SM#C':'2010',
    '@RP#C':'2002',
    '@LE#C':'2005',
    '@SP#C':'2010',
    '@MME#C':'2010',
    '@S#C':'2005',
    'LG#C':'2010',
} # bugged data
DONTLIKE = [
#     'GAS#C', # NG trades weird
#     'QNG#C', # NG trades weird
#     '@QG#C', # NG trades weird
    '@ED#C', #we want to stay further out the ED curve
    'EZ#C',
    '@TU#C',
]

START_DATE = '2000-1-1'

def fix_fut(fut):
    for column,date in BROKEN.items():
        if column in fut.columns:\
            fut[column] = fut[column][date:]
        
    return fut

def load_symbols_and_prices(sectors):
    symbols = pd.read_csv(DATA_DIR / 'symbols.csv', index_col='iqsymbol')

    symbols_list = [
        (x, Path(DATA_DIR / 'daily' / '%s.csv' % x)) for x in symbols.index
        if (symbols.loc[x]['Sector'] in sectors)
    ]
    symbols_list = [(symbol,pp) for symbol, pp in symbols_list if pp.exists() and pp.size > 10000 and not symbol in DONTLIKE and not symbol in BROKEN]
    symbols_list, fns = zip(*symbols_list)
    dfs = [pd.read_csv(ff, index_col='date', parse_dates=True)[['close_p']] for ff in fns]
    fut = pd.concat([x['close_p'] for x in dfs], axis=1).ffill().truncate(before=pd.Timestamp(START_DATE))
    fut.columns = symbols_list

    fut = fix_fut(fut)
#     print(len(fut.columns))
#     print(fut.columns)
    return symbols, fut

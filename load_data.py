import pandas as pd
from path import Path

DATA_DIR = Path('../data/')
DAILY_DATA_DIR = DATA_DIR / 'daily'

def load_contract(contract_name, directory):
    series = pd.read_csv(DATA_DIR / directory / '{}.csv'.format(contract_name), index_col=0)
    series = series[::-1]
    if directory == 'minutely':
        series['Time'] = series['date'] + ' ' + series['time']
        series = series.set_index(pd.to_datetime(series['Time'], format='%Y-%m-%d 0 days %H:%M:00.000000000'))
    else:
        series['Time'] = series['date']
        series = series.set_index(pd.to_datetime(series['Time'], format='%Y-%m-%d'))
    
    series = series[['open_p', 'close_p', 'prd_vlm', 'Time']]
    series = series.rename(columns={'close_p':'Close', 'open_p':'Open', 'prd_vlm':'Volume'})
    series['Instrument'] = contract_name
    return series

def load_contracts(symbol, directory='minutely'):
    contract_names = [x.basename().namebase for x in (DATA_DIR / directory).files('*{}*'.format(symbol))]
    loaded = [load_contract(x, directory) for x in contract_names]
    loaded = list(sorted(loaded, key=lambda x:x.index[-1]))
    first = loaded[0]
    # cut out from later contracts what former contracts already have
    zipped = zip(loaded, loaded[1:])
    cut_contracts = [latter.truncate(before=former.index[-1] + pd.Timedelta(minutes=1)) for former, latter in zipped]

    return pd.concat([first] + cut_contracts)

def load_all_cont_contracts():
    all_continuous_contracts = DAILY_DATA_DIR.files('*#C*')
    all_continuous_contracts = [x.basename().namebase for x in all_continuous_contracts]
    return {name: load_contract(name, 'daily') for name in all_continuous_contracts}
import pandas as pd

def non_negative_series(series):
    series = series.copy(deep=True)
    series['Returns'] = series['Close'].diff() / series['Close'].shift(1)
    series['rPrices'] = (1 + series['Returns']).cumprod()
    return series

def daily_bars(series):
    series = series.copy(deep=True)
    return group_bars(series, series.index.date)

def volume_bars(series, bar_size=10000):
    series = series.copy(deep=True)
    series['Cum Volume'] = series['Volume'].cumsum()
    bar_idx = (series['Cum Volume'] / bar_size).round(0).astype(int).values
    return group_bars(series, bar_idx)

def dollar_bars(series, bar_size=10000 * 3000):
    series = series.copy(deep=True)
    series['Dollar Volume'] = (series['Volume'] * series['Close'])
    series['Cum Dollar Volume'] = series['Dollar Volume'].cumsum()
    bar_idx = (series['Cum Dollar Volume'] / bar_size).round(0).astype(int).values
    return group_bars(series, bar_idx)

def group_bars(series, bar_idx):
    gg = series.groupby(bar_idx)
    df = pd.DataFrame()
    df['Volume'] = gg['Volume'].sum()
    if 'Dollar Volume' in series.columns:
        df['Dollar Volume'] = gg['Dollar Volume'].sum()
    df['Open'] = gg['Open'].first()
    df['Close'] = gg['Close'].last()
    if 'rPrices' in series.columns:
        df['rPrices'] = gg['rPrices'].last()
    df['Instrument'] = gg['Instrument'].first()
    df['Time'] = gg.apply(lambda x:x.index[0])
    df['Num Ticks'] = gg.size()
    df = df.set_index(gg.apply(lambda x:x.index[0]))
    return df

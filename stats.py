# from https://github.com/esvhd/pypbo/blob/master/pypbo/pbo.py

import scipy.stats as ss
import numpy as np
def psr(sharpe, T, skew, kurtosis, target_sharpe=0):
    """
    Probabilistic Sharpe Ratio.
    Parameters:
        sharpe:
            observed sharpe ratio, in same frequency as T.
        T:
            no. of observations, should match return / sharpe sampling period.
        skew:
            sharpe ratio skew
        kurtosis:
            sharpe ratio kurtosis
        target_sharpe:
            target sharpe ratio
    Returns:
        Cumulative probabilities for observed sharpe ratios under standard
        Normal distribution.
    """
    value = (
        (sharpe - target_sharpe)
        * np.sqrt(T - 1)
        / np.sqrt(1.0 - skew * sharpe + sharpe ** 2 * (kurtosis - 1) / 4.0)
    )
    psr = ss.norm.cdf(value, 0, 1)
    return psr

def expected_max(N):
    """
    Expected maximum of IID random variance X_n ~ Z, n = 1,...,N,
    where Z is the CDF of the standard Normal distribution,
    E[MAX_n] = E[max{x_n}]. Computed for a large N.
    """
    if N < 5:
        raise AssertionError("Condition N >> 1 not satisfied.")
    return (1 - np.euler_gamma) * ss.norm.ppf(
        1 - 1.0 / N
    ) + np.euler_gamma * ss.norm.ppf(1 - np.exp(-1) / N)


def dsr(test_sharpe, sharpe_std, N, T, skew, kurtosis):
    """
    Deflated Sharpe Ratio statistic. DSR = PSR(SR_0).
    See paper for definition of SR_0. http://ssrn.com/abstract=2460551
    Parameters:
        test_sharpe :
            reported sharpe, to be tested.
        sharpe_std :
            standard deviation of sharpe ratios from N trials / configurations
        N :
            number of backtest configurations
        T :
            number of observations
        skew :
            skew of returns
        kurtosis :
            kurtosis of returns
    Returns:
        DSR statistic
    """
    # sharpe_std = np.std(sharpe_n, ddof=1)
    target_sharpe = sharpe_std * expected_max(N)

    dsr_stat = psr(test_sharpe, T, skew, kurtosis, target_sharpe)

    return dsr_stat

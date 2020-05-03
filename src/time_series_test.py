import statsmodels.tsa.api as smt
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pandas import Series


class TimeSeriesTest:

    @staticmethod
    def tsplot(y, lags=None, figsize=(12, 7), syle='bmh'):
    
        if not isinstance(y, Series):
            y = Series(y)
        
        with plt.style.context(style='bmh'):
            fig = plt.figure(figsize=figsize)
            layout = (2,2)
            ts_ax = plt.subplot2grid(layout, (0,0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (1,0))
            pacf_ax = plt.subplot2grid(layout, (1,1))
            
            y.plot(ax=ts_ax)
            p_value = sm.tsa.stattools.adfuller(y)[1]
            ts_ax.set_title(f'Time Series Analysis Plots\n Dickey-Fuller: p={round(p_value, 5)}')
            smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
            smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
            plt.tight_layout()
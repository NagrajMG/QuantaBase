from capm import CAPM
from markowitz import MarkowitzModel
from regimeSwitching import RegimeSwitch

if __name__ == '__main__':
    start_date = '2020-01-01'
    end_date = '2025-01-01'
    stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']

    # # Capital Asset Pricing Model
    capm = CAPM(['AAPL', '^GSPC'], start_date, end_date)
    capm.initialize()
    capm.calculate_beta()
    capm.regression()

    # Markowitz Model
    NUM_PORTFOLIOS = 10000
    model = MarkowitzModel(start_date, end_date, stocks, NUM_PORTFOLIOS)
    model.runMarkowitz()

    # Regime switching Model
    switchModel = RegimeSwitch(start_date, end_date, stocks)
    weights_0, weights_1 = switchModel.runModel()
    switchModel.computeMetrics(weights_0, weights_1)

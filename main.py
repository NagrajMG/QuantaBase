from capm import CAPM
from markowitz import MarkowitzModel
if __name__ == '__main__':
    start_date = '2015-01-01'
    end_date = '2025-01-01'

    # Capital Asset Pricing model
    capm = CAPM(['AAPL', '^GSPC'], start_date, end_date)
    capm.initialize()
    capm.calculate_beta()
    capm.regression()

    # stocks handled for Markowitz model
    stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']
    # Generate random weights (different portfolios)
    NUM_PORTFOLIOS = 25000  
    model = MarkowitzModel(start_date, end_date, stocks, NUM_PORTFOLIOS)
    model.runMarkowitz()

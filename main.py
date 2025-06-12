from capm import CAPM
from markowitz import MarkowitzModel
from regimeSwitching import RegimeSwitch
from BlackScholes import OptionPricing

if __name__ == '__main__':
    start_date = '2020-01-01'
    end_date = '2025-01-01'
    stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']

    # # # Capital Asset Pricing Model
    # capm = CAPM(['AAPL', '^GSPC'], start_date, end_date)
    # capm.initialize()
    # capm.calculate_beta()
    # capm.regression()

    # # Markowitz Model
    # NUM_PORTFOLIOS = 10000
    # Markowitz = MarkowitzModel(start_date, end_date, stocks, NUM_PORTFOLIOS)
    # Markowitz.runMarkowitz()

    # # Regime switching Model
    # switchModel = RegimeSwitch(start_date, end_date, stocks)
    # weights_0, weights_1 = switchModel.runModel()
    # switchModel.computeMetrics(weights_0, weights_1)

    # Black-Scholes Option Pricing
    S0=100					#underlying stock price at t=0
    E=100					#strike price
    T = 1					#expiry
    rf = 0.05				#risk-free rate
    sigma=0.2				#volatility of the underlying stock
    iterations = 1000000	#number of iterations in the Monte-Carlo simulation	

    blackScholes = OptionPricing(S0,E,T,rf,sigma,iterations)
    print("Call option price with Monte-Carlo approach: ", blackScholes.call_option_simulation()) 
    print("Put option price with Monte-Carlo approach: ", blackScholes.put_option_simulation())




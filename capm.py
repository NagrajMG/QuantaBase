from utils import *

# market interest rate
RISK_FREE_RATE = 0.05
# calculate the annual return
MONTHS_IN_YEAR = 12

class CAPM:

    def __init__(self, stocks, start_date, end_date):
        self.data = None
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self):
    # dictionary with keys as stock names
       data = {}

       for stock in self.stocks:
        # closing prices
           ticker = yf.Ticker(stock)
           data[stock] = ticker.history(start=self.start_date, end=self.end_date)['Close']
    
       return (pd.DataFrame(data))

    def initialize(self):
        stock_data = self.download_data()
        # Monthly returns
        stock_data = stock_data.resample('ME').last()

        self.data = pd.DataFrame({'s_close': stock_data[self.stocks[0]],
                                  'm_close': stock_data[self.stocks[1]]})

        # logarithmic monthly returns: Stock returns, market returns 
        self.data[['s_returns', 'm_returns']] = np.log(self.data[['s_close', 'm_close']] /
                                                       self.data[['s_close', 'm_close']].shift(1))

        # remove the NaN values: First value removed
        self.data = self.data[1:]

    def calculate_beta(self):
        # covariance matrix: the diagonal items are the variances
        # off diagonals are the covariances
        covariance_matrix = np.cov(self.data["s_returns"], self.data["m_returns"])
        # calculating beta according to the formula
        beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
        print("Beta from formula: ", beta)

    def regression(self):
        # using linear regression to fit a line to the data
        # [stock_returns, market_returns] - slope is the beta
        beta, alpha = np.polyfit(self.data['m_returns'], self.data['s_returns'], deg=1)
        print("Beta from regression: ", beta)
        # calculate the expected return according to the CAPM formula

        # we are after annual return (this is why multiply by 12)
        expected_return = RISK_FREE_RATE + beta * (self.data['m_returns'].mean()*MONTHS_IN_YEAR - RISK_FREE_RATE)
        print("Expected return: ", expected_return)
        self.plot_regression(alpha, beta)

    def plot_regression(self, alpha, beta):
        fig, axis = plt.subplots(1, figsize=(20, 10))
        axis.scatter(self.data["m_returns"], self.data['s_returns'],label="Data Points")
        axis.plot(self.data["m_returns"], beta * self.data["m_returns"] + alpha, color='red', label="CAPM Line")
        plt.title('Capital Asset Pricing Model, finding alpha and beta')
        plt.xlabel('Market return $R_m$', fontsize=18)
        plt.ylabel('Stock return $R_a$')
        plt.text(0.08, 0.05, r'$R_a = \beta * R_m + \alpha$', fontsize=18)
        plt.legend()
        plt.grid(True)
        plt.savefig("Plots/beta_for_capm.png")
        
    





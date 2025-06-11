from utils import *

# Trading days
NUM_TRADING_DAYS = 252

class MarkowitzModel:
    def __init__(self, start_date, end_date, stocks, NUM_PORTFOLIOS):
        self.start_date = start_date
        self.end_date = end_date
        self.stocks = stocks
        self.NUM_PORTFOLIOS = NUM_PORTFOLIOS

    def download_data(self):
       stock_data = {}

       for stock in self.stocks:
            # closing prices
           ticker = yf.Ticker(stock)
           stock_data[stock] = ticker.history(start=self.start_date, end=self.end_date)['Close']
       
       return (pd.DataFrame(stock_data))


    def show_data(self, data):
        data.plot(figsize=(10, 5))
        plt.show()


    def calculate_return(self, data):
        # NORMALIZATION - to measure all variables in comparable metric
       self.log_return = np.log(data / data.shift(1))
    #    print(self.log_return[1:]) 
       return self.log_return


    def show_statistics(self, returns):
    # instead of daily metrics we are after annual metrics
    # mean of annual return
       print(returns.mean() * NUM_TRADING_DAYS)
       print(returns.cov() * NUM_TRADING_DAYS)


    def show_mean_variance(self, returns, weights):
        # we are after the annual return
       portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
       portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()* NUM_TRADING_DAYS, weights)))
       print("Expected portfolio mean (return): ", portfolio_return)
       print("Expected portfolio volatility (standard deviation): ", portfolio_volatility)
    
    def show_portfolios(self, returns, volatilities):
        plt.figure(figsize=(10, 6))
        plt.scatter(volatilities, returns, c=returns / volatilities, marker='o')
        plt.grid(True)
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.colorbar(label='Sharpe Ratio')
        plt.savefig("Plots/portfolios_randomized.png")


    def generate_portfolios(self):
        portfolio_means = []
        portfolio_risks = []
        portfolio_weights = []

        for _ in range(self.NUM_PORTFOLIOS):
            w = np.random.random(len(self.stocks))
            w /= np.sum(w)
            portfolio_weights.append(w)
            portfolio_means.append(np.sum(self.log_return.mean() * w) * NUM_TRADING_DAYS)
            portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(self.log_return.cov() * NUM_TRADING_DAYS, w))))
        
        return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)


    def statistics(self, weights, returns):
        portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()* NUM_TRADING_DAYS, weights)))
        return np.array([portfolio_return, portfolio_volatility, portfolio_return / portfolio_volatility])


    # scipy optimize module can find the minimum of a given function
    # the maximum of a f(x) is the minimum of -f(x)
    def min_function_sharpe(self, weights, returns):
        return -self.statistics(weights, returns)[2]


    # what are the constraints? The sum of weights = 1 !!!
    # f(x)=0 this is the function to minimize
    def optimize_portfolio(self, weights):
        # the sum of weights is 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # the weights can be 1 at most: 1 when 100% of money is invested into a single stock
        bounds = tuple((0, 1) for _ in range(len(self.stocks)))
        return optimization.minimize(fun=self.min_function_sharpe, x0=weights[0], args=self.log_return, method='SLSQP', bounds=bounds, constraints=constraints)


    def print_optimal_portfolio(self, optimum):
        print("Optimal portfolio: ", optimum['x'].round(3))
        print("Expected return, volatility and Sharpe ratio: ",self.statistics(optimum['x'].round(3), self.log_return))


    def show_optimal_portfolio(self,opt, portfolio_rets, portfolio_vols):
        plt.figure(figsize=(10, 6))
        plt.scatter(portfolio_vols, portfolio_rets, c=portfolio_rets / portfolio_vols, marker='o')
        plt.grid(True)
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.colorbar(label='Sharpe Ratio')
        plt.plot(self.statistics(opt['x'], self.log_return)[1], self.statistics(opt['x'], self.log_return)[0], 'g*', markersize=20.0)
        plt.savefig('Plots/optimal_portfolio.png')

    def runMarkowitz(self):
       dataset = self.download_data()
       log_returns = self.calculate_return(dataset)
       weights, means, risks = self.generate_portfolios()
       self.show_portfolios(means, risks)
       optimum = self.optimize_portfolio(weights)
       self.print_optimal_portfolio(optimum)
       self.show_optimal_portfolio(optimum,means, risks )
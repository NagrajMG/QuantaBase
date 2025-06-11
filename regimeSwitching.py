from utils import *

# Trading days
NUM_TRADING_DAYS = 252

class RegimeSwitch:
    def __init__(self, start_date, end_date, stocks):
        self.start_date = start_date
        self.end_date = end_date
        self.stocks = stocks

    def download_data(self):
       stock_data = {}
       for stock in self.stocks:
           # closing prices
           ticker = yf.Ticker(stock)
           stock_data[stock] = ticker.history(start=self.start_date, end=self.end_date)['Close']
       return (pd.DataFrame(stock_data))

    def calculate_return(self, data):
        # NORMALIZATION - to measure all variables in comparable metric
       self.log_return = np.log(data / data.shift(1))
       self.log_return = self.log_return[1:]
       return self.log_return
    
    def regimePrediction(self):
        X = self.log_return.values
        print("Any NaN:", np.isnan(X).any())
        print("Any Inf:", np.isinf(X).any())
        print("Min, Max:", np.min(X), np.max(X))
        sequences = [X]
        num_states = 2
        distributions = [Normal() for _ in range(num_states)]
        model = DenseHMM(distributions, verbose=True)
        # 4. Fit the model to your data (Baum-Welch algorithm)
        model.fit(sequences)
        probs = model.predict_proba(sequences)
        probs = probs.squeeze()   
        hidden_states = torch.argmax(probs, dim=1).numpy()
        self.log_return['Regime'] = hidden_states
        # print(self.log_return)
        plt.figure(figsize=(12, 4))
        plt.plot(self.log_return.index, hidden_states, label='Regime')
        plt.title("Inferred Market Regimes")
        plt.savefig('Plots/predicted_regimes.png')
    
    def RegimeSplitting(self):
        # Split by regime
        returns_0 = self.log_return[self.log_return['Regime'] == 0].drop(columns='Regime')
        returns_1 = self.log_return[self.log_return['Regime'] == 1].drop(columns='Regime')
        # Regime-wise stats
        mu_0 = returns_0.mean().values
        cov_0 = returns_0.cov().values

        mu_1 = returns_1.mean().values
        cov_1 = returns_1.cov().values
        return mu_0, cov_0, mu_1, cov_1
    
    def optimizeWeights(self, mu, cov, gamma=0.5):
        n = len(mu)
        w = cp.Variable(n)
        obj = cp.Maximize(mu @ w - gamma * cp.quad_form(w, cov))
        constraints = [
        cp.sum(w) == 1,
        w >= 0
        ]
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return w.value
    

    def runModel(self):
        dataset = self.download_data()
        returns_df = self.calculate_return(dataset)
        self.regimePrediction()
        mu_0, cov_0, mu_1, cov_1 = self.RegimeSplitting()
        w_0 = self.optimizeWeights(mu_0, cov_0)
        w_1 = self.optimizeWeights(mu_1, cov_1)
        return w_0, w_1
    
    def computeMetrics(self, weights_0, weights_1):
        regimes = self.log_return["Regime"].values
        returns_matrix = self.log_return.drop(columns=["Regime"]).to_numpy()
        # Step 3: Regime masks
        regime_mask_0 = (regimes == 0)
        regime_mask_1 = (regimes == 1) 
        # Step 4: Allocate portfolio returns
        portfolio_returns = np.zeros(len(regimes))
        portfolio_returns[regime_mask_0] = np.sum(returns_matrix[regime_mask_0] * weights_0, axis=1)
        portfolio_returns[regime_mask_1] = np.sum(returns_matrix[regime_mask_1] * weights_1, axis=1)
        avg_return = np.mean(portfolio_returns)
        volatility = np.std(portfolio_returns)
        sharpe_ratio = (avg_return) / volatility * np.sqrt(252)
        cumulative_return = np.exp(np.sum(portfolio_returns)) - 1
        drawdowns = 1 - np.exp(np.cumsum(portfolio_returns)) / np.maximum.accumulate(np.exp(np.cumsum(portfolio_returns)))
        max_drawdown = np.max(drawdowns)

        print("The Sharpe ratio for portfolio using current weights: ",sharpe_ratio )
        print("Max Drawdown: ", max_drawdown)
    

    


    
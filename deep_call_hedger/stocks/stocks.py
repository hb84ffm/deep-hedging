# stocks.py
import numpy as np

class Stocks:
    """Simulate stock price paths using Geometric Brownian Motion with pathwise sampled drift, volatility & strikes."""

    def __init__(self, 
                 paths: int, 
                 timesteps: int, 
                 T: float, 
                 mu_min: float, 
                 mu_max: float, 
                 sigma_min: float, 
                 sigma_max: float, 
                 S0_min: float, 
                 S0_max: float):
        
        self.paths = paths
        self.timesteps = timesteps
        self.T = T
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.S0_min = S0_min
        self.S0_max = S0_max

    def dt(self):
        """Return step size for time."""
        return self.T / (self.timesteps - 1)

    def K(self, S0):
        """Sample uniformly distributed strikes per path."""
        return np.random.uniform(0.8 * S0, 1.2 * S0, self.paths)

    def mu(self):
        """Sample uniformly drift per path."""
        return np.random.uniform(self.mu_min, self.mu_max, self.paths)

    def sigma(self):
        """Sample uniformly volatilities per path."""
        return np.random.uniform(self.sigma_min, self.sigma_max, self.paths)

    def Zt(self):
        """Sample random normal increments per path & timestep."""
        return np.random.randn(self.paths, self.timesteps - 1)

    def St(self, mu, sigma, dt, Zt):
        """Generate simulated stock price paths following a GBM."""
        stocks = np.zeros((self.paths, self.timesteps))
        # Start prices
        stocks[:, 0] = np.random.uniform(self.S0_min, self.S0_max, self.paths)
        # Time evolution
        for t in range(1, self.timesteps):
            stocks[:, t] = (stocks[:, t - 1]* np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Zt[:, t - 1]))
        return stocks

    def simulate(self):
        """Create full stock price simulation & return all relevant variables."""
        dt = self.dt()
        mu = self.mu()
        sigma = self.sigma()
        Zt = self.Zt()
        stocks = self.St(mu, sigma, dt, Zt)
        K = self.K(stocks[:, 0])
        return dt, mu, sigma, Zt, stocks, K, self.T

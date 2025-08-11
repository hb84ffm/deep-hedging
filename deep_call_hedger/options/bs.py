# bs.py
import numpy as np
import scipy.stats as ss

class BS:
    """Black Scholes option pricing for calls with pathwise inputs."""

    @staticmethod
    def call_delta_d1_d2(S, K, mu, sigma, T, t):
        """Black Scholes formula for European call price, delta, d1, d2 at any timestep.
        Assumption mu = r (risk-neutral measure)."""
        tau = T - t
        if tau == 0:  # At maturity
            call = max(S - K, 0)
            delta = np.where(S > K, 1, 0)
            d1 = d2 = np.nan
            return call, delta, d1, d2
        else:
            d1 = (np.log(S / K) + (mu + sigma**2 / 2) * tau) / (sigma * np.sqrt(tau))
            d2 = d1 - sigma * np.sqrt(tau)
            delta = ss.norm.cdf(d1)
            call = S * delta - K * np.exp(-mu * tau) * ss.norm.cdf(d2)
            return call, delta, d1, d2

    def __init__(self, stocks, K, T, mu, sigma):
        self.stocks = stocks
        self.K = K
        self.paths, self.timesteps = stocks.shape # read paths & timesteps from stocks, since options correspond! 
        self.T = T
        self.mu = mu
        self.sigma = sigma
        self.dt = T / (self.timesteps - 1)

    def call_deltas_d1s_d2s(self):
        """Apply Black Scholes calculation to all paths/timesteps.
        Return calls, deltas, d1s, d2s as 2D NumPy arrays."""
        calls = np.zeros_like(self.stocks, dtype=float)
        deltas = np.zeros_like(self.stocks, dtype=float)
        d1s = np.zeros_like(self.stocks, dtype=float)
        d2s = np.zeros_like(self.stocks, dtype=float)

        for j in range(self.paths):
            for i in range(self.timesteps):
                t = i * self.dt
                call, delta, d1, d2 = BS.call_delta_d1_d2(self.stocks[j, i], self.K[j], self.mu[j], self.sigma[j], self.T, t)
                calls[j, i] = call
                deltas[j, i] = delta
                d1s[j, i] = d1
                d2s[j, i] = d2
        return calls, deltas, d1s, d2s

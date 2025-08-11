#prediction.py
import numpy as np
from tensorflow.keras.models import load_model
from deep_call_hedger.stocks.stocks import Stocks
from deep_call_hedger.options.bs import BS
from deep_call_hedger.dh_model.dh_model import ZeroDeltaLayer, SliceLayer, PayoffLayer


class Prediction:
    """Runs predictions using the deep hedging model & computes the Black Scholes benchmark."""

    def __init__(self,
                 nr_of_paths,
                 nr_of_timesteps,
                 T,
                 mu_min,
                 mu_max,
                 sigma_min,
                 sigma_max,
                 S0_min,
                 S0_max,
                 model_path): # important: here the path to the used trained model must be stated!
        
        self.nr_of_paths = nr_of_paths
        self.nr_of_timesteps = nr_of_timesteps
        self.T = T
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.S0_min = S0_min
        self.S0_max = S0_max
        self.model_path = model_path

    def create_prediction_data(self):
        """Simulate stock and option paths using classes Stocks (stocks.py) and BS (bs.py) ."""
        stock_sim = Stocks(paths=self.nr_of_paths,
                           timesteps=self.nr_of_timesteps + 1, # we need one more stock price for payoff calculation
                           T=self.T,
                           mu_min=self.mu_min,
                           mu_max=self.mu_max,
                           sigma_min=self.sigma_min,
                           sigma_max=self.sigma_max,
                           S0_min=self.S0_min,
                           S0_max=self.S0_max)
        dt, mu, sigma, Zt, stocks, K, T = stock_sim.simulate()

        bs_calc = BS(stocks=stocks, K=K, T=T, mu=mu, sigma=sigma)
        calls, deltas, d1s, d2s = bs_calc.call_deltas_d1s_d2s() # Notice: deltas here are the Black Scholes deltas, the network deltas are named nn_deltas!
        return stocks, K, calls, deltas

    def prepare_inputs(self, stocks, calls, K):
        """Prepare model inputs, all stocks, call prices at t=0 & strikes K."""
        C0 = calls[:, 0:1]
        inputs = [stocks.astype(np.float32),C0.astype(np.float32),K.astype(np.float32)]
        return inputs

    def predict(self):
        """Load trained model using the path "model_path", simulate prediction data, run prediction & compute some statistics."""
        model = load_model(self.model_path,custom_objects={'ZeroDeltaLayer': ZeroDeltaLayer,
                                                           'SliceLayer': SliceLayer,
                                                           'PayoffLayer': PayoffLayer})
        stocks, K, calls, deltas = self.create_prediction_data()
        inputs = self.prepare_inputs(stocks, calls, K)
        run_prediction = model.predict(inputs)
        nn_final_pnl, nn_deltas, nn_pnl_t = run_prediction

        bs_pnls = np.zeros((self.nr_of_paths, self.nr_of_timesteps + 2)) # shapes: stocks (paths, steps+1), deltas (paths, steps+1), nn_deltas (paths, steps)
        bs_pnls[:, 0:1] = calls[:, 0:1]

        # Calculate Black Scholes pnl (all timesteps, all paths)
        for i in range(self.nr_of_timesteps):
            bs_pnls[:, i + 1] = bs_pnls[:, i] + deltas[:, i] * (stocks[:, i + 1] - stocks[:, i])
        bs_pnls[:, -1] = bs_pnls[:, self.nr_of_timesteps] - np.maximum(stocks[:, -1] - K, 0)

        # Calculate Model pnl (all timesteps, all paths)
        nn_pnls = np.concatenate([calls[:, 0:1], nn_pnl_t], axis=1)

        # Statistics
        nn_final_pnl_avg = np.round(np.mean(nn_final_pnl), 2)
        nn_final_pnl_stddev = np.round(np.std(nn_final_pnl), 2)
        bs_final_pnl_avg = np.round(np.mean(bs_pnls[:, -1]), 2)
        bs_final_pnl_stddev = np.round(np.std(bs_pnls[:, -1]), 2)

        return stocks, K, calls, deltas, nn_final_pnl, nn_deltas, bs_pnls, nn_pnls,nn_final_pnl_avg, nn_final_pnl_stddev, bs_final_pnl_avg, bs_final_pnl_stddev,run_prediction

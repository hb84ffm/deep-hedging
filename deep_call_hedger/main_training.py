# main_training.py
from deep_call_hedger.stocks import stocks
from deep_call_hedger.options import bs
from deep_call_hedger.dh_model import dh_model
from deep_call_hedger.training import training

class Main:
    """Main class to orchestrate simulation, model creation, and training."""

    def __init__(
        self, 
        T: int, 
        nr_of_timesteps: int,
        nr_of_paths: int,
        S0_min: float,
        S0_max: float,
        mu_min: float,
        mu_max: float,
        sigma_min: float,
        sigma_max: float,
        nr_of_layers: int,
        nr_of_neurons: list[int],
        activations: list[str],
        weights_mu: list[float],
        weights_sigma: list[float],
        bias_mu: list[float],
        bias_sigma: list[float],
        batch_size: int,
        epochs: int,
        optimizer: str,
        loss: str,
        validation_split: float):

        self.T = T
        self.nr_of_timesteps = nr_of_timesteps
        self.nr_of_paths = nr_of_paths
        self.S0_min = S0_min
        self.S0_max = S0_max
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.nr_of_layers = nr_of_layers
        self.nr_of_neurons = nr_of_neurons
        self.activations = activations
        self.weights_mu = weights_mu
        self.weights_sigma = weights_sigma
        self.bias_mu = bias_mu
        self.bias_sigma = bias_sigma
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.validation_split = validation_split

    def run(self):
        """Run stocks, option pricing, model creation & training pipeline."""
        stock_simulator = stocks.Stocks(
            paths=self.nr_of_paths,
            timesteps=self.nr_of_timesteps + 1,  # we need one more price for payoff calculation
            T=self.T,
            mu_min=self.mu_min,
            mu_max=self.mu_max,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            S0_min=self.S0_min,
            S0_max=self.S0_max)
        dt, mu, sigma, Zt, simulated_stocks, K, T = stock_simulator.simulate()
        self.simulated_stocks = simulated_stocks
        self.K = K

        option_simulator = bs.BS(
            stocks=simulated_stocks,
            K=K,
            T=T,
            mu=mu,
            sigma=sigma)
        calls, deltas, d1s, d2s = option_simulator.call_deltas_d1s_d2s()
        self.calls = calls
        self.bs_deltas = deltas

        run_model = dh_model.DHModel(
            n=self.nr_of_timesteps,
            nr_of_layers=self.nr_of_layers,
            nr_of_neurons=self.nr_of_neurons,
            activations=self.activations,
            weights_mu=self.weights_mu,
            weights_sigma=self.weights_sigma,
            bias_mu=self.bias_mu,
            bias_sigma=self.bias_sigma)
        model = run_model.create_model()
        self.model = model

        run_training = training.Training(
            nr_of_paths=self.nr_of_paths,
            model=model,
            stocks=simulated_stocks,
            calls=calls,
            K=K,
            batch_size=self.batch_size,
            epochs=self.epochs,
            optimizer=self.optimizer,
            loss=self.loss,
            validation_split=self.validation_split)
        trained_model = run_training.start_training()

        self.model.save('deep_hedging_64.keras') # name carries "64", since its trained on 64 fixed timesteps! If adjustments are done make sure to rename to choosen timesteps.
        return trained_model, self.model

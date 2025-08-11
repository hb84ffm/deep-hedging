# main_prediction.py
from deep_call_hedger.prediction import prediction
from deep_call_hedger.prediction import analysis

class Main:
    """Main class to orchestrate prediction and creation of charts & metrics for analysis."""

    def __init__(self,
                 nr_of_paths: int,
                 nr_of_timesteps: int,
                 T: float,
                 mu_min: float,
                 mu_max: float,
                 sigma_min: float,
                 sigma_max: float,
                 S0_min: float,
                 S0_max: float,
                 model_path: str,
                 path_to_analyse: int):
        
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
        self.path_to_analyse = path_to_analyse

    def _plot(self, path_to_analyse: int):
        """Internal method to create charts for a given path index."""
        run_plot = analysis.Analysis(
            stocks=self.stocks,
            K=self.K,
            calls=self.calls,
            deltas=self.deltas,
            nn_final_pnl=self.nn_final_pnl,
            nn_deltas=self.nn_deltas,
            bs_pnls=self.bs_pnls,
            nn_pnls=self.nn_pnls,
            nn_final_pnl_avg=self.nn_final_pnl_avg,
            bs_final_pnl_avg=self.bs_final_pnl_avg,
            nn_final_pnl_stddev=self.nn_final_pnl_stddev,
            bs_final_pnl_stddev=self.bs_final_pnl_stddev,
            path_to_analyse=path_to_analyse)
        
        run_plot.create_charts()

    def run(self):
        """Run the prediction once, unpack results, then plot charts for initial path."""
        predict_now = prediction.Prediction(
            nr_of_paths=self.nr_of_paths,
            nr_of_timesteps=self.nr_of_timesteps,
            T=self.T,
            mu_min=self.mu_min,
            mu_max=self.mu_max,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            S0_min=self.S0_min,
            S0_max=self.S0_max,
            model_path=self.model_path)
        
        stocks, K, calls, deltas, nn_final_pnl, nn_deltas, bs_pnls, nn_pnls, nn_final_pnl_avg, \
        nn_final_pnl_stddev, bs_final_pnl_avg,bs_final_pnl_stddev, run_prediction = predict_now.predict()

        # transform unpacked variables to instance variables
        self.stocks = stocks
        self.deltas = deltas
        self.K = K
        self.calls = calls
        self.nn_final_pnl = nn_final_pnl
        self.nn_deltas = nn_deltas
        self.bs_pnls = bs_pnls
        self.nn_pnls = nn_pnls
        self.nn_final_pnl_avg = nn_final_pnl_avg
        self.nn_final_pnl_stddev = nn_final_pnl_stddev
        self.bs_final_pnl_avg = bs_final_pnl_avg
        self.bs_final_pnl_stddev = bs_final_pnl_stddev
        self.run_prediction = run_prediction

        self._plot(self.path_to_analyse)

    def analyse_path(self, path_to_analyse: int):
        """Plot charts for a given path index without having to rerun prediction (run()) again"""
        self.path_to_analyse = path_to_analyse
        self._plot(path_to_analyse)

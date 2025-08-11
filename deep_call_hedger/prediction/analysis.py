#analysis.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")


class Analysis:
    """Class that helps the user to analyse teh results, providng some charts of hedging results."""

    def __init__(self,
                 stocks,
                 K,
                 calls,
                 deltas,  # Black Scholes deltas
                 nn_final_pnl,  # final value from nn_pnl_t (last timestep value)
                 nn_deltas,  # network deltas
                 bs_pnls,
                 nn_pnls,
                 nn_final_pnl_avg,
                 bs_final_pnl_avg,
                 nn_final_pnl_stddev,
                 bs_final_pnl_stddev,
                 path_to_analyse):
        
        self.stocks = stocks
        self.K = K
        self.calls = calls
        self.deltas = deltas
        self.nn_final_pnl = nn_final_pnl
        self.nn_deltas = nn_deltas
        self.bs_pnls = bs_pnls
        self.nn_pnls = nn_pnls
        self.nr_of_paths = stocks.shape[0]
        self.nr_of_timesteps = stocks.shape[1] - 1
        self.nn_final_pnl_avg = nn_final_pnl_avg
        self.bs_final_pnl_avg = bs_final_pnl_avg
        self.nn_final_pnl_stddev = nn_final_pnl_stddev
        self.bs_final_pnl_stddev = bs_final_pnl_stddev
        self.path_to_analyse = path_to_analyse

    def plot_stock_call(self):
        """Plot stock, call prices & strike for the choosen path."""
        plt.figure(figsize=(8, 2))
        plt.plot(range(self.stocks.shape[1]),
                 self.stocks[self.path_to_analyse, :],
                 label='stock', marker='.', markersize=3, linewidth=0.8, color="#0D982B")
        plt.plot(range(self.calls.shape[1]),
                 self.calls[self.path_to_analyse, :],
                 label='call', marker='.', markersize=3, linewidth=0.8, color="#F06406")
        plt.plot(range(self.calls.shape[1]),
                 self.K[self.path_to_analyse] * np.ones(self.calls.shape[1]),
                 label='strike', color="#000000", linestyle="-", linewidth=0.5)  # thinner strike line
        plt.xlabel('Timestep', fontsize=8)
        plt.ylabel('Price', fontsize=8)
        plt.title(f'Comparison of stock, call (Black Scholes) & strike (path={self.path_to_analyse})', fontsize=9)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.legend(fontsize=7)
        plt.show()

    def plot_deltas(self):
        """Plot model vs Black Scholes deltas for the chosen path."""
        plt.figure(figsize=(8, 2))
        plt.plot(range(self.nn_deltas.shape[1]),
                 self.nn_deltas[self.path_to_analyse, :],
                 label='model', marker='.', markersize=3, linewidth=0.8, color="#0C0058")
        plt.plot(range(self.nn_deltas.shape[1]),
                 self.deltas[self.path_to_analyse, :-1],
                 label='bs', marker='.', markersize=3, linewidth=0.8, color="#F06406")
        plt.xlabel('Timestep', fontsize=8)
        plt.ylabel('Delta', fontsize=8)
        plt.title(f'Delta comparison (Black Scholes VS Model, path={self.path_to_analyse})', fontsize=9)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.legend(fontsize=7)
        plt.show()

    def plot_pnls(self):
        """Plot model vs Black-Scholes PnLs for the chosen path."""
        plt.figure(figsize=(8, 2))
        plt.plot(range(self.nr_of_timesteps + 2),
                 self.nn_pnls[self.path_to_analyse, :],
                 label='model', marker='.', markersize=3, linewidth=0.8, color="#0C0058")
        plt.plot(range(self.nr_of_timesteps + 2),
                 self.bs_pnls[self.path_to_analyse, :],
                 label='bs', marker='.', markersize=3, linewidth=0.8, color="#F06406")
        plt.xlabel('Timestep', fontsize=8)
        plt.ylabel('PnL', fontsize=8)
        plt.title(f'PnL comparison (Black Scholes VS Model, path={self.path_to_analyse})', fontsize=9)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.legend(fontsize=7)
        plt.show()

    def plot_final_pnls_sorted(self):
        """Plot sorted final pnls comparing model and Black Scholes."""
        plt.figure(figsize=(8, 2))
        plt.plot(np.arange(len(np.sort(self.nn_final_pnl.flatten()))),
                 np.sort(self.nn_final_pnl.flatten()),
                 label='model', marker='.', markersize=3, linewidth=0.8, color="#0C0058")
        plt.plot(np.arange(len(np.sort(self.nn_final_pnl.flatten()))),
                 np.sort(self.bs_pnls[:, -1]),
                 label='bs', marker='.', markersize=3, linewidth=0.8, color="#F06406")
        plt.xlabel('Path (sorted by pnl)', fontsize=8)
        plt.ylabel('Final PnL', fontsize=8)
        plt.title('Final PnL sorted (Black Scholes VS Model, all paths)', fontsize=9)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.legend(fontsize=7)
        plt.show()

    def plot_histogram(self):
        """Plot histogram of final pnls comparing model and Black Scholes."""
        plt.figure(figsize=(8, 2))
        plt.hist(self.nn_final_pnl.flatten(),
                 bins=50,
                 label=f'model, avg={self.nn_final_pnl_avg:.2f}, st_dev={self.nn_final_pnl_stddev:.2f}',
                 color="#0C0058",
                 edgecolor='white',
                 linewidth=0.2)
        plt.hist(self.bs_pnls[:, -1],
                 bins=50,
                 label=f'bs, avg={self.bs_final_pnl_avg:.2f}, st_dev={self.bs_final_pnl_stddev:.2f}',
                 color="#F06406",
                 edgecolor='white',
                 linewidth=0.2)
        plt.xlabel('Final PnL', fontsize=8)
        plt.ylabel('Observations', fontsize=8)
        plt.title('Histogram of final PnL (Black Scholes VS Model, all paths)', fontsize=9)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.legend(fontsize=7)
        plt.show()

    def create_charts(self):
        """Generate all charts."""
        self.plot_stock_call()
        self.plot_deltas()
        self.plot_pnls()
        self.plot_final_pnls_sorted()
        self.plot_histogram()

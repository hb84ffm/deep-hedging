# dh_model.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model as md
from tensorflow.keras.layers import Input, Dense, Concatenate, Add, Subtract, Multiply
from tensorflow.keras import initializers

class ZeroDeltaLayer(tf.keras.layers.Layer):
    """Custom layer returning zeros like the first column slice of input."""
    def call(self, x):
        return tf.zeros_like(x[:, 0:1])

    def get_config(self):
        return super().get_config()

class SliceLayer(tf.keras.layers.Layer):
    """Custom layer to slice input tensor at index t."""
    def __init__(self, t, **kwargs):
        super().__init__(**kwargs)
        self.t = t

    def call(self, x):
        return x[:, self.t : self.t + 1]

    def get_config(self):
        config = super().get_config()
        config.update({"t": self.t})
        return config

class PayoffLayer(tf.keras.layers.Layer):
    """Custom layer to calculate payoff max(S_T - K_strike, 0)."""
    def call(self, inputs):
        S_term, K_strike = inputs
        return tf.maximum(S_term - K_strike, 0)

    def get_config(self):
        return super().get_config()

class DHModel:
    """Deep hedging model encapsulating the computational graph."""

    def __init__(self,
                 n,
                 nr_of_layers,
                 nr_of_neurons,
                 activations,
                 weights_mu,
                 weights_sigma,
                 bias_mu,
                 bias_sigma):
        
        self.n = n  # timesteps
        self.nr_of_layers = nr_of_layers  # number of layers in a network (fixed architecture across all timesteps)
        self.nr_of_neurons = nr_of_neurons  # neurons per layer
        self.activations = activations  # activation functions between layers
        self.weights_mu = weights_mu  # means to initialize weights
        self.weights_sigma = weights_sigma  # std.dev to initialize weights
        self.bias_mu = bias_mu  # mean to initialize biases
        self.bias_sigma = bias_sigma  # std.dev to initialize biases

    def create_model(self):
        """Create deep hedging computational graph using Keras functional API."""
# ------------------------------------------------------------------------  START OF COMPUTATIONAL GRAPH ------------------------------------------------------------------------  
        # Inputs
        S = Input(shape=(self.n + 1,), name="S")  # stock price input
        C0 = Input(shape=(1,), name="C0")         # initial call price input t=o
        K_strike = Input(shape=(1,), name="K")    # strike price input

        # Define layer names dynmaically
        if self.nr_of_layers > 1:
            layer_names = ["layer_hidden"] * (self.nr_of_layers - 1) + ["layer_output"]
        else:
            layer_names = ["layer_output"]

        # Prepare weighted dense layers, one set per timestep
        layers = []
        for t in range(self.n):
            for i in range(self.nr_of_layers):
                layer = Dense(
                    self.nr_of_neurons[i],
                    activation=self.activations[i],
                    kernel_initializer=initializers.RandomNormal(self.weights_mu[i], self.weights_sigma[i]),
                    bias_initializer=initializers.RandomNormal(self.bias_mu[i], self.bias_sigma[i]),
                    name=f"{layer_names[i]}_{i}_{t}")
                layers.append(layer)

# t=0 (left boundary)
        delta_prev = ZeroDeltaLayer()(S)
        pnl = C0

# t=1,...,n-1 (inner iterations)
        nn_deltas = [] # to store network deltas per timestep
        nn_pnl_t = [] # store cumulative pnl per timestep
        for t in range(self.n):
            S_now = SliceLayer(t)(S) # replaces Lambda slicing for S[:, t:t+1] & gets current stock price S_t
            features = Concatenate()([S_now, delta_prev]) # features=(S_t, delta_{t-1})

            h = features
            for i in range(self.nr_of_layers - 1):
                # inner layers run until i=n-1 #
                # for t=0 sequence is 0,1,...,nr_of_layers-1
                # for t=1 sequence is nr_of_layers,...,2*nr_of_layers-1 
                # ...  
                # above sequence can then be generalized by t*nr_of_layers+i with i=0,...nr_of_layers-1 for any arbitrary t
                idx = t * self.nr_of_layers + i 
                h = layers[idx](h) 

            idx = t * self.nr_of_layers + (self.nr_of_layers - 1) # output layer for each t
            delta_now = layers[idx](h) # predicted delta at time t
            nn_deltas.append(delta_now)

            S_next = SliceLayer(t + 1)(S) # replaces Lambda slicing for S[:, t+1:t+2] & gets next stock price S_t+1
            S_increment = Subtract()([S_next, S_now]) # S_{t+1} - S_t
            pnl_increment = Multiply()([delta_now, S_increment]) # delta_t * (S_{t+1} - S_t)
            pnl = Add()([pnl, pnl_increment]) # Pnl_t=Pnl_t-1+delta_t*(S_t+1-S_t)
            nn_pnl_t.append(pnl) # store new pnl value in list
            delta_prev = delta_now # set current network delta to delta_prev (that is network input at next iteration!)

# t=n (right boundary)
        S_terminal = SliceLayer(self.n)(S) # get last stock price S_n
        payoff = PayoffLayer(name="payoff")([S_terminal, K_strike])  # calculate payoff max(S_n - K, 0)
        pnl_final = Subtract()([pnl, payoff]) # final pnl = cumulative pnl - payoff
        nn_pnl_t.append(pnl_final) # store final pnl in list

        nn_deltas_concat = Concatenate(name="nn_deltas")(nn_deltas) # must concat list of tensors into single tensor for output, otherwise we get error
        nn_pnl_t_concat = Concatenate(axis=1, name="nn_pnl_t")(nn_pnl_t)
# ------------------------------------------------------------------------  END OF COMPUTATIONAL GRAPH ------------------------------------------------------------------------
        model = md(inputs=[S, C0, K_strike],outputs=[pnl_final, nn_deltas_concat, nn_pnl_t_concat],name="deep_hedging")

        return model

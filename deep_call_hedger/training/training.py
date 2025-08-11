# training.py
import numpy as np
import tensorflow as tf
from tensorflow import keras

class Training:
    """Runs the training for the deep hedging model."""

    def __init__(self, 
                 nr_of_paths: int, 
                 model, 
                 stocks: list, 
                 calls: list, 
                 K: list,
                 batch_size: int, 
                 epochs: int, 
                 optimizer: str, 
                 loss: str, 
                 validation_split: float):
        
        self.nr_of_paths = nr_of_paths
        self.model = model
        self.stocks = stocks
        self.calls = calls
        self.K = K
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.validation_split = validation_split

    def start_training(self):
        """ Define nr_of_paths, model features, labels, batch size, epochs, optimizer, loss & validation split.
            then compile the model, start training, output loss & return the fit history."""
        features = [self.stocks.astype(np.float32),
            self.calls[:, 0:1].astype(np.float32),
            self.K.reshape(-1, 1).astype(np.float32)]

        # notice: keras requires lables for each model output (we have nn_pnl_final & nn_deltas), even if an output does not contribute to loss! 
        labels_final_pnl = np.zeros((self.nr_of_paths, 1), dtype=np.float32) # define labels (in our case 0)
        labels_nn_deltas = np.zeros((self.nr_of_paths, self.stocks.shape[1] - 1),dtype=np.float32) # dummy labels for deltas to satisify keras input requirements
        labels_nn_pnl_t = np.zeros((self.nr_of_paths, self.stocks.shape[1]),dtype=np.float32)
        labels = [labels_final_pnl, labels_nn_deltas, labels_nn_pnl_t]

        self.model.compile(optimizer=self.optimizer,
            loss=[self.loss, None, None],  # set loss for "real" labels to self.loss & for not needed dummy values to None
            loss_weights=[1, 0, 0]) # set loss weights for "real" labels to 1 & for not needed dummy values to 0

        trained_model = self.model.fit(
            x=features,
            y=labels,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=True)

        return trained_model

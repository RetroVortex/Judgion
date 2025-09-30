# Iván Ontiveros - RetroVortex


# This script implements the Grid Search algorithm to find the best hyperparamethers for building the neural networks
# The user can change the "PARAM_GRID" global variable to choose the range of the hyperparamethers that will be used during the search
# The execution returns the best combination of hyperparamethers found based on the accuracy of its predictions


# Libraries
from tensorflow import keras
from keras.optimizers import Adam, SGD, AdamW
import numpy as np
import pandas as pd
import time
from functools import partial
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from scikeras.wrappers import KerasClassifier

from judgionLib.utils import stats_getter, normalize_stats, model_builder
from judgionLib.constants import TRAINING_DIRECTORY, USE_SEED, TRAINING_SEED, LOG_DIRECTORY

# Some optimizers can cause problems when using Grid Search, like AdamW. To solve them, add this code for said optimizer:
from keras.utils import get_custom_objects
get_custom_objects().update({'AdamW': AdamW})


# TODO: Global hyperparamethers grid. You need to change the values in this dictionary in order to conduct the Grid Search.
# This is the version of the variable that was last used in order to find the best paramethers for "The Pyramid"
PARAM_GRID = {
            "model__hidden_layers": [
                [
                    {"neurons": 360, "activation": "sigmoid", "input_shape": (46,)},
                    {"neurons": 160, "activation": "gelu"},
                    {"neurons": 45, "activation": "relu"},
                ]
            ],
            "model__optimizer": [partial(AdamW, learning_rate=0.003, beta_1=0.95, beta_2=0.999, weight_decay=0.00003, epsilon = 0.0001),
                                 partial(AdamW, learning_rate=0.003, beta_1=0.95, beta_2=0.999, weight_decay=0.003, epsilon = 0.0001),
                                 partial(AdamW, learning_rate=0.003, beta_1=0.95, beta_2=0.999, weight_decay=0.03, epsilon = 0.0001),
                                 partial(AdamW, learning_rate=0.003, beta_1=0.95, beta_2=0.999, weight_decay=0.3, epsilon = 0.0001),
                                 partial(AdamW, learning_rate=0.003, beta_1=0.95, beta_2=0.999, weight_decay=3, epsilon = 0.0001),
                                ],
            "batch_size": [16],
            "epochs": [80, 130]
        }

class PARAMS_FINDER:

    def __init__(self):

        # Training matrices
        self.x_train = []       # Round stats
        self.y_train = []       # Round winners

        # This flag will be up if the user chooses to normalize the data for the training process
        self.norm_flag = False


    # This method implements the search process
    def start_search(self):

        # Normalize the data if the user chose to
        if self.norm_flag:
            normalized_stats, _ = normalize_stats(self.x_train)
            x_train_processed = np.array(normalized_stats)
        else:
            x_train_processed = np.array(self.x_train)

        # Transforming 'y_train' into a one-hot encoded array
        y_train_processed = np.array(self.y_train)
        
        # We define the model wrapper with 'EarlyStopping', so the training will stop if the model stops learning, making the search more efficient
        model_wrapper = KerasClassifier(model=model_builder, loss="categorical_crossentropy", verbose=0, validation_split=0.1, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)])
        #model_wrapper = KerasClassifier(model=model_builder, verbose=0, validation_split=0.1, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)])
        
        # Define the grid of hyperparameters to search
        param_grid = PARAM_GRID
        
        # 3-fold cross validation will help with the class imbalance
        strat = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # Defining the Grid Search
        grid = GridSearchCV(estimator=model_wrapper, param_grid=param_grid, cv=strat, scoring="accuracy", n_jobs=-1, verbose=2, error_score='raise')
        
        # Start timing the grid search
        start_time = time.time()
        grid_result = grid.fit(x_train_processed, y_train_processed)
        end_time = time.time()

        # Store the results in a CSV file
        cv_res = pd.DataFrame(grid.cv_results_)
        cv_res.to_csv(f"{LOG_DIRECTORY}/grid_log_{int(time.time())}.csv", index=False)
        
        # Output the training time, best score, and best parameters
        print("Grid search training time = {:.2f} s".format(end_time - start_time))
        print("Best accuracy: {:.4f} using parameters: {}".format(grid_result.best_score_, grid_result.best_params_))
        
        return grid_result


    # This function initializes the class variables and starts the paramethers searching process
    def init_search(self):

        # Initialize the training matrices
        (self.x_train, self.y_train) = stats_getter(TRAINING_DIRECTORY)

        # Normalize the stats
        x = input("Normalize stats for training? [Y/N]: ")
        if x == 'Y' or x == 'y':
            self.norm_flag = True

        # Start the paramethers search
        self.start_search()


# Main
if __name__ == '__main__':
    if (USE_SEED):
        print("FIXED TRAINING SEED SET")
        keras.utils.set_random_seed(TRAINING_SEED)
        
    params_finder = PARAMS_FINDER()
    params_finder.init_search()
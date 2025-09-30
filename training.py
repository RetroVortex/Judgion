# Iván Ontiveros - RetroVortex


# This script uses the dataset generated in order to train machine learning models
# When trained, the user will have the option to store the judge
# Right now, no testing is implemented; the judge is manually tested with the "round_judge" script


# Libraries
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD, AdamW
from keras.utils import to_categorical
import time
import numpy as np
import csv

from judgionLib.utils import stats_getter, normalize_stats, model_builder, DiagonalLayer
from judgionLib.constants import TRAINING_DIRECTORY, MODELS_DIRECTORY, USE_SEED, TRAINING_SEED, LOG_DIRECTORY, STAT_MAP


class AI_UFC_TRAINER:

    def __init__(self):
        
        # Training matrices
        self.x_train = []   # Round stats (training data)
        self.y_train = []   # Round winners (labels)

        # This flag will be up if the user chooses to normalize the data for the training process
        self.norm_flag = False

   
    # Main training method. This is the exact version that was used to train "The Pyramid"
    def training_process(self):

        # Specifying the neural network's structure
        hidden_layers = [
                    {"neurons": 360, "activation": "sigmoid", "input_shape": (46,)},
                    {"neurons": 160, "activation": "gelu"},
                    {"neurons": 45, "activation": "relu"},
                ]
        opti = AdamW(learning_rate=0.003, beta_1=0.95, beta_2=0.999, weight_decay=0.01, epsilon=0.00001)

        # Compiling the model
        judge = model_builder(hidden_layers, opti)

        # As this solution has 4 output neurons and only one will be True (1) for each round,
        # transform 'y_train' into a one-hot encoded array
        y_train_nn = to_categorical(self.y_train, num_classes=4)
        # And the 'x_train' will be a Numpy array, normalized or raw depending on the user's choice
        if self.norm_flag:
            normalized_stats, highest_values = normalize_stats(self.x_train)
            x_train_nn = np.array(normalized_stats)
        else:
            x_train_nn = np.array(self.x_train)

        # Start the training process, monitorising the time it consumes
        ini_t = time.time()
        _ = judge.fit(x_train_nn, y_train_nn, batch_size=16, epochs=330, validation_split=0.0, shuffle=True)
        end_t = time.time()
        print("Training time = ", end_t-ini_t, " s")

        # Ask the user if the judge model should be saved
        answer = input("Save judge? [Y/N]: ")
        if answer == 'Y' or answer == 'y':
            name = input("Enter the judge's name: ")
            judge.save(f'{MODELS_DIRECTORY}/{name}.h5')
            # If we trained the model with normalized data, store the highest_values array in a 'npy' file
            if self.norm_flag:
                np.save(f'{MODELS_DIRECTORY}/{name}.npy', highest_values)


    # This method implements the training process using the Diagonal Layer, just as it was used to train "The Diagonal"
    def training_process_diagonal_layer(self):
        
        # Init a Sequential Neural Network
        judge = Sequential()
        # The first hidden layer is a Diagonal Layer, which individually multiplies each stat with their associated weight
        judge.add(DiagonalLayer(input_shape=(46,)))
        # The output layer will have 4 neurons (one for every possible outcome), with the softmax activation function
        judge.add(Dense(4, activation='softmax'))

        # As this solution has 4 output neurons and only one will be True (1) for each round,
        # transform 'y_train' into a one-hot encoded array
        y_train_nn = to_categorical(self.y_train, num_classes=4)

        # We will use the raw data for the input Numpy array
        x_train_nn = np.array(self.x_train)

        # Adam optimizer
        opt = Adam(learning_rate=0.00175, beta_1=0.97, beta_2=0.999, weight_decay = 0.0002)

        # We'll compile the model using:
        # Categorical crossentropy as the loss function
        # The optimizer specified earlier
        # 'Accuracy' as the metric
        judge.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # Start the training process, monitorising the time it consumes
        ini_t = time.time()
        fitting = judge.fit(x_train_nn, y_train_nn, batch_size=4, epochs=350, validation_split=0.0, shuffle=True)
        end_t = time.time()
        print("Training time = ", end_t - ini_t, " s")

        # Saving the weights in the log directory
        diag_weights, _ = judge.layers[0].get_weights()
        diag_weights = diag_weights.tolist()
        #print("DiagonalLayer - weights:", diag_weights)
        stat_labels = [entry['name'] for entry in STAT_MAP[:23]] * 2
        log_path = f"{LOG_DIRECTORY}/DiagonalLayerWeights_{int(time.time())}.csv"
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(stat_labels)
            writer.writerow(diag_weights)

        print(f"Saved diagonal weights to {log_path}")

        # Ask user if the trained model should be saved
        answer = input("Save judge? [Y/N]: ")
        if answer.lower() == 'y':
            name = input("Enter the judge's name: ")
            judge.save(f'{MODELS_DIRECTORY}/{name}.h5')


    # This function initializes the class variables and starts the training process
    def init_training(self):

        # Initialize the training matrices
        (self.x_train, self.y_train) = stats_getter(TRAINING_DIRECTORY)
        #print("Number of rounds: ", len(self.y_train))

        # Normalize the stats
        x = input("Normalize stats for training? [Y/N]: ")
        if x == 'Y' or x == 'y':
            self.norm_flag = True

        # Diagonal Layer training
        y = input("Diagonal Layer training? [Y/N]: ")
        if y == 'Y' or y == 'y':
            self.training_process_diagonal_layer()

        else:
            # Start the training
            self.training_process()



# Main
if __name__ == '__main__':
    if (USE_SEED):
        print("FIXED TRAINING SEED SET")
        keras.utils.set_random_seed(TRAINING_SEED)

    trainer = AI_UFC_TRAINER()
    trainer.init_training()
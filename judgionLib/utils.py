# judgionLib/utils.py
# Iván Ontiveros - RetroVortex


# This script includes the methods that are used by more than one class in the project


import os
import json
from tensorflow import keras
from keras.layers import Layer, Dense, AlphaDropout
from keras.models import Sequential
from keras import backend as K


# This recursive function extract the numbers stored in a dictionary in order, returning them as an array
# This function only works properly with Python version >=3.7
def stats_extractor(data):
    numbers = []

    # Iterate over the dictionary
    for key, value in data.items():

        # If the value is another dictionary, make a recursive call
        if isinstance(value, dict):
            numbers.extend(stats_extractor(value))

        # If the value is a number, add it to the list
        elif isinstance(value, (int, float)):
            numbers.append(value)

    return numbers


# This function returns the winner of a round if its stats were inversed
def parallel_winner(winner):
    
    # If the red corner fighter wins a round 10-8 (winner = 0), return 3 (8-10 for blue corner)
    if winner == 0:
        return 3
    # If the red corner fighter wins a round 10-9 (winner = 1), return 2 (9-10 for blue corner)
    elif winner == 1:
        return 2
    
    # Same logic applies for rounds won by the blue corner
    elif winner == 2:
        return 1
    return 0


# This function reads the JSON files from the dataset and stores the information in the training matrices
def stats_getter(directory):
    stats = []
    winners = []

    # Iterate over the all the JSON files within the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:

                # Load the data stored inside a file
                data = json.load(file)
                
                # Open the first layer of dictionaries in different variables
                red_stats = data['red_fighter']
                blue_stats = data['blue_fighter']
                winner = data['winner']
                
                # Read the dictionaries for both fighters, storing the values in arrays of integers
                red = stats_extractor(red_stats)
                blue = stats_extractor(blue_stats)

                # First, the round is stored as it is
                fight_stats = []
                fight_stats.extend(red)
                fight_stats.extend(blue)
                stats.append(fight_stats)
                # The label will also be stored as the original
                winners.append(winner)

                # Then, store another round, symmetric to the original one
                fight_stats = []
                fight_stats.extend(blue)
                fight_stats.extend(red)
                stats.append(fight_stats)
                # Invert the label, and store it
                winners.append(parallel_winner(winner))

    return stats, winners


# This function finds the highest values for each stat in the dataset, returning them in an array
def highest_values_finder(stats):

    # We'll check the first half of all rows in the matrix (as the rounds are reverse-duplicated)
    stats_num = len(stats[0]) // 2      # 23
    highest_values = [0.0] * stats_num
    
    for i in range(stats_num):

        # Control time highest value is 300 seconds
        if i == 22:
            highest_values[i] = 300

        # For the other stats, we check which is the higher value in the whole dataset
        else:
            for j in range(len(stats)):
                # If our current value is higher than the one stored, we store the current value as the new max value
                if stats[j][i] > highest_values[i]:
                    highest_values[i] = stats[j][i]
                    
            # Max value shouldn't be 0, but it could be if you aren't using the 'cuts' stat
            # That's why I'm adding this check, we will divide by 1 if all found values for a stat are 0
            if highest_values[i] == 0:
                highest_values[i] = 1.0
                
    return highest_values


# This method can be used to normalize the training data, so its values are in the range [0, 1]
def normalize_stats(stats):
    
    # We need to find the highest value for each stat
    highest_values = highest_values_finder(stats)
    
    # Duplicating the array for simplifying coding the normalizing part
    highest_values_full = highest_values + highest_values

    # Normalising the matrix by dividing each stat with the highest values in the dataset
    normalized_stats = []
    for row in stats:
        new_row = []
        for i, value in enumerate(row):
            normalized_value = value / highest_values_full[i]
            new_row.append(normalized_value)
        normalized_stats.append(new_row)

    return normalized_stats, highest_values_full


# This method returns a kernel initializer depending on the activation function received as paramether
def get_kernel(activation):
    
    a = activation.lower()

    # If you use an activation function not present here, "glorot_uniform" will be used; add more if needed
    if a == "selu":
        return "lecun_normal"
    if a in ("relu", "elu"):
        return "he_normal"
    if a in ("sigmoid", "tanh", "gelu", "silu"):
        return "glorot_uniform"
    
    return "glorot_uniform"


# This method is the neural network builder, returning a compiled model using the paramethers specified in its call
def model_builder(hidden_layers, optimizer):

    K.clear_session()

    # Initialize the Sequential model
    model = Sequential()

    # Iterate through each layer defined in the structure and add to the model.
    for i, layer in enumerate(hidden_layers):
        neurons = layer['neurons']
        activation = layer['activation']
        kwargs = {}
        # For the first layer, we also include the input layer
        if i == 0 and "input_shape" in layer:
            kwargs["input_shape"] = layer['input_shape']

        # Extract the Kernel initializer specified by the user. If None, we default it depending on the activation function using the "get_kernel" method.
        kernel_func = layer.get("kernel_initializer", get_kernel(activation))

        model.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_func, **kwargs))

        # Alpha Dropout paramether, must be > 0.0 to activate it
        if "alpha_dropout" in layer:
            dropout_rate = layer['alpha_dropout']
            if dropout_rate > 0.0:
                model.add(AlphaDropout(dropout_rate))


    # The last layer will always have 4 neurons and the 'softmax' activation function
    model.add(Dense(4, activation='softmax'))

    # Instantiate the optimizer with the given learning rate
    opt = optimizer() if callable(optimizer) else optimizer

    # Compile the model specifying categorical crossentropy as the loss and accuracy as the metric
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model


# Class for the Diagonal Layer, a layer which only has one weight per stat, therefore making a "linear transformation" to the data
class DiagonalLayer(Layer):

    def __init__(self, **kwargs):
        super(DiagonalLayer, self).__init__(**kwargs)

    # This method builds the layer structure, initializing the weights and biases
    def build(self, input_shape):

        # Weight vector, initialized as 1s. This vector is trainable, so its values will change during the training process
        self.diag = self.add_weight(name='diag',
                                    shape=(input_shape[-1],),
                                    initializer='ones',
                                    trainable=True)

        # Bias vector, initialized as 1s. This vector is not trainable, so its values won't change during the training process
        self.bias = self.add_weight(name='bias',
                                    shape=(input_shape[-1],),
                                    initializer='zeros',
                                    trainable=False)  # Set to True if you want to make the bias trainable
                                    
        super(DiagonalLayer, self).build(input_shape)

    # This method implements the layer's operation
    def call(self, inputs):

        # Stat * Weight + Bias
        return inputs * self.diag + self.bias   # Since 'bias' will be 0, this equals "stat * weight". If you decide to make the bias trainable, you won't need to change this.
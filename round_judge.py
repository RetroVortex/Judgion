# Iván Ontiveros - RetroVortex


# This script takes a file with a trained judge model and use it for judging new rounds
# Instantiations of this class are used in the main script. Each judge there is initialized as an object of the AI_JUDGE class


# Libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
import numpy as np
import json
import os

from judgionLib.utils import stats_extractor, DiagonalLayer
from judgionLib.constants import TEST_DIRECTORY, MODELS_DIRECTORY


class AI_JUDGE:

    def __init__(self):

        # This variable will store a judge model generated with the training script
        self.judge_model = None

        # This variable will store the ID for the fight to score
        self.fight_id = None

        # This variable is used for flow control
        self.keep_going = True

        # These variables store the names of the fighters which competed in the round that we are scoring
        self.red_fighter = None
        self.blue_fighter = None

        # This array stores the normalization factors for this judge's training data (if normalization was applied)
        self.norm_factors = None    # TODO: Buscar en el directorio de modelos si hay un '.npy' aparte del '.h5', y meterlo aquí y usarlo para el judgind process si tal


    # Function to load the stats for the round selected
    def load_data(self, filename):

        # Opening the file
        with open(filename, 'r') as file:
            data = json.load(file)
        
        # Isolate the stats
        red_stats = data['red_fighter']
        blue_stats = data['blue_fighter']

        # Store the last name of each fighter
        self.red_fighter = red_stats['name']
        self.blue_fighter = blue_stats['name']

        # Extract the stats and store them in matrices
        red = stats_extractor(red_stats)
        blue = stats_extractor(blue_stats)

        # Prepare the model input by combining red and blue stats
        model_input = []
        model_input.extend(red)
        model_input.extend(blue)

        # Return the matrix rearranged so it fits the Keras structure (1x46)
        return np.array(model_input).reshape(1,-1)


    # This method returns all the rounds that match the ID specified by the user in a sorted list
    def rounds_extractor(self):
        fight_rounds = []

        # Iterate over the directory and extract all the files that match the ID
        for filename in os.listdir(TEST_DIRECTORY):
            if filename.startswith(self.fight_id) and filename.endswith('.json'):
                fight_rounds.append(filename)

        # Sorting the rounds (R1 first, then R2, and so on)
        fight_rounds.sort(key=lambda x: int(x.split('_R')[-1].replace('.json', '')))
        
        return fight_rounds


    # This function processes the model's output and prints it in a friendly/readable way
    def print_probabilities(self, prediction):

        # First, create the 4 possible output keys
        keys = [f"10-8 {self.red_fighter}", f"10-9 {self.red_fighter}", f"10-9 {self.blue_fighter}", f"10-8 {self.blue_fighter}"]

        # Process each value
        values = []
        for value in prediction[0]:
            processed_value = round(float(value) * 100, 2)  # Probability in percentage and with two decimals
            if processed_value < 0.01:  
                processed_value = 0  # Assume a 0 probability if the output is too small
            values.append(processed_value)

        # Associate each processed value with its key
        probabilities = zip(keys, values)

        # Sort the probabilities from higher to lower
        sorted_probabilities = sorted(probabilities, key=lambda x: x[1], reverse=True)

        # Print the processed probabilities in order
        print("---- Probabilities ----")
        for label, value in sorted_probabilities:
            print(f"{label}: {value:.2f}%")


    # This function executes the judging process
    def give_scorecards(self):

        # Extracting the files that match the fight specified by the user
        fight_files = self.rounds_extractor()

        # If none are found, end the judging process
        if not fight_files:
            print(f"There are no fights that match the fight ID ({self.fight_id}).\n")
            print("Enter a valid fight ID and restart the process.\n")
            return

        # Initial scores for each fighter
        red_score = 0
        blue_score = 0

        # Loop through the directory
        for filename in fight_files:
                
            print("OFFICIAL SCORECARD")
            
            # Extract the round number (assuming the format: {RedFighterLastName}_{BlueFighterLastName}_R{round number}.json)
            round_number = filename.replace(self.fight_id, '').replace('.json', '').strip('_R')

            # Load the data for the current round
            file_path = os.path.join(TEST_DIRECTORY, filename)
            input_data = self.load_data(file_path)

            # print("Input data: ", input_data)

            # Checking if the data should be normalized, and doing so if needed
            if self.norm_factors is not None:
                input_data = input_data / self.norm_factors
                # print("Normalized data: ", input_data)

            # Get the prediction
            prediction = self.judge_model.predict(input_data)

            # Print the round number
            print("")
            print("")
            print(f"ROUND {round_number}")
            print("-----------")
            print("")

            # Full prediction output
            self.print_probabilities(prediction)

            # Winner of the round (summary)
            predicted_class = np.argmax(prediction)
            print("")
            print("     OFFICIAL RESULT")
            print("")
            if predicted_class == 0:
                print("Red corner | 10 | 8 | Blue corner")
                red_score += 10
                blue_score += 8
            elif predicted_class == 1:
                print("Red corner | 10 | 9 | Blue corner")
                red_score += 10
                blue_score += 9
            elif predicted_class == 2:
                print("Red corner | 9 | 10 | Blue corner")
                red_score += 9
                blue_score += 10
            else:
                print("Red corner | 8 | 10 | Blue corner")
                red_score += 8
                blue_score += 10

            # Separator for the next round
            print("")
            print('================================================')
            print('================================================')
            print("")


        # Print the final results
        print("")
        print("----------------------------------------")
        print("|   Fight results (Official scorecard)   |")
        print("----------------------------------------")
        print("")
        print("Red corner | ", red_score, " | ", blue_score, " | Blue corner")
        print("")
        if red_score > blue_score:
            print("Winner: ", self.red_fighter)
            return 1
        elif blue_score > red_score:
            print("Winner: ", self.blue_fighter)
            return 2
        else:
            print("Result: Draw")
            return 0

    
    # This function sets which judge model will be used
    def set_judge(self):
        models = []

        # Loop through the current directory
        for filename in os.listdir(MODELS_DIRECTORY):

            # Get all the judge models
            if filename.endswith('.h5'):
                models.append(f'{MODELS_DIRECTORY}/{filename}')

        # List available models (if there are any)
        if not models:
            print("No models available!")
            return None

        print("\nList of available judges: \n")

        for i, model_name in enumerate(models):
            judge_name = model_name.replace(f'{MODELS_DIRECTORY}/','').replace('_', ' ').replace('.h5', '')
            print(f"{i + 1}. {judge_name}")

        # Ask the user to pick a model
        while True:
            choice = input("\nSelect the model number: ")

            # Check if the pick is valid. If not, ask again
            if choice.isdigit() and 1 <= int(choice) <= len(models):
                # Load and return the chosen model
                chosen_model = models[int(choice) - 1]

                with custom_object_scope({'DiagonalLayer': DiagonalLayer}):     # This line is only used for compatibility with a personalised layer
                    self.judge_model = load_model(chosen_model)

                # Checking if there's any '.npy' file assigned to this judge
                base_name = os.path.splitext(os.path.basename(chosen_model))[0]
                npy_path = os.path.join(MODELS_DIRECTORY, base_name + '.npy')
                # If there is, it means we need to normalize the data before getting the prediction
                if os.path.exists(npy_path):
                    self.norm_factors = np.load(npy_path)
                    print(f"Normalization factors for the selected judge: {self.norm_factors}")

                return

            else:
                print("Invalid choice, please select again.")
        

    # This function sets the fight ID
    def set_id(self):

        print("")
        r_ln = input("Enter the red corner fighter (full) last name: ").strip()
        print("")
        b_ln = input("Enter the blue corner fighter (full) last name: ").strip()
        print("")
        return (f'{r_ln}_{b_ln}')


    # This function starts the judging process
    def start(self):

        # First, we set the judge model
        self.set_judge()    # After the call, the class variable 'judge_model' will store the AI model

        # If no model was available/selected, end the execution
        if self.judge_model is None:
            print("No model selected, exiting.")
            return

        while self.keep_going:
            # We set the fight ID
            self.fight_id = self.set_id()

            # Then, we start the judging process
            _ = self.give_scorecards()

            # Ask the user if he wants to continue judging
            cont = input("Judge another fight? (y/n): ")
            # If not, end execution
            if cont == 'n':
                self.keep_going = False


# Main
if __name__ == '__main__':
    judge = AI_JUDGE()
    judge.start()
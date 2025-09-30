# Judgion

## Relevant information

The current release is a **beta** version (labeled as `v0.0.0`). Judgion will be actively updated; the first planned milestone is **January 2026**.

Live scorecards during UFC events will be posted on X (Twitter) and Instagram:
- X: [@judgion](https://x.com/judgion)
- Instagram: [@judgionmma](https://instagram.com/judgionmma)

When citing this work, please reference the repository as shown in the `CITATION.cff`. You can contact me through social media or via email at: [judgion@gmail.com](mailto:judgion@gmail.com).


## Installation

I suggest using a Conda/Miniconda environment for Judgion. After downloading the repository, open a CMD in the directory where you installed Judgion and follow these steps:

- Create a new environment with Python and pip:

```cmd
conda create -n {NEW_ENV_NAME} python=3.8 pip -y
```

- Activate it:

```cmd
conda activate {NEW_ENV_NAME}
```

- Install the requirements:

```cmd
python -m pip install -r requirements.txt
```

## Scripts

### Main

The main script implements the judging process for a fight. It assumes you already have trained models (at least the ones featured in Judgion). 

1. Choose which models will be used for the judging. Just enter the number associated with the judge in each list. 

2. You will be asked for the link to the stats page of the UFC fight. Copy and paste it on the terminal and press Enter,

3. One JSON file per round will be generated in the 'test' directory. You can modify it if necessary; when all is set, press Enter.

4. All scorecards have been generated and a winner has been declared. You can check the round-by-round scoring in the terminal.

### Json_Generator

This script is used for generating JSON files out of real UFC fights. You can run it with the flags "-r" (to remove the last round, useful if the fight was stopped) and "-t" (to generate the JSON files in the 'test' directory instead of the 'training' one). To use it:

1. Look for the UFC stats page for the fight you want to scrape and paste it in the "LINK_UFC_WEBSITE" global variable you'll find at the top of the script.

2. Run the script; the JSON files are now generated.

### Round_Judge

This script is useful for testing a single model. It assumes you already have generated the JSON files for the round/fight you want to score (in the 'test' directory!). To use it:

1. Choose a model by writing the number associated with it and pressing Enter.

2. Write the name of the red corner fighter, and then the name of the blue corner fighter. 

3. All found rounds with both fighters involved are scored using the chosen judge.

### Training

The training script generated a new model using the JSON files stored in the 'training' directory as the dataset. You can change the "training_process" method as you wish in order to train your own models. You can also use the input normalization that was done to train The Decayed and The Pyramid models. You can also use the personalised Diagonal Layer used for both The Diagonal and The Cross judges.

### Params_Finder

This script implements Grid Search in order to find the best combination of both architecture and hyperparamethers to achieve the best possible performance in a model. To use it, just change the "PARAM_GRID" global variable you'll find at the top of the script, writing the variables you want to apply Grid Search to. Each execution generates a CSV file which is stored in the 'log' directory.

### Data_Visualizer

The last script can be used to generate graphs out of the generated dataset. It's useful in order to improve the understanding of the dataset while building it, keeping track of the statistics and trying to balance the data. All generated graphs for the dataset used when training Judgion models are included in the "graphs" directory.

## Additional files

- The **jugdion-dataset** is the directory where training files are stored. I've included the 32 theoretical rounds I used in order to train Judgion models.

- In the **models** directory, all six Judgion models are included. There are two 'npy' files, they are arrays which store the highest values for all statistics in the training dataset, in order to apply the same input normalization to the new inputs that the one that was applied during their training.

- In the **graphs** directory, I've included all the graphs generated for the training dataset that I used to train Judgion models.

- In the **logs** directory, CSV files generated during the execution of Grid Search will be stored. I've included an example so you can check the structure of the file.

- The **judgionLib** directory acts as a library where I include global variables and methods that are used multiple times in the repository, in different scripts.
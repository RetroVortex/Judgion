# judgionLib/constants.py
# Iván Ontiveros - RetroVortex

# Paths/Directories
TRAINING_DIRECTORY = "judgion-dataset"
TEST_DIRECTORY = 'test'
MODELS_DIRECTORY = 'models'
LOG_DIRECTORY = 'logs'

# Number of judges for the main script judging process
NUM_JUDGES = 3

# Training seed. Useful for replicating the model training with the best hyperparamethers found with Grid Search
USE_SEED = False
TRAINING_SEED = 120

# General stat map used in data_visualizer
STAT_MAP = [
    {'stat_index': 0, 'name': 'Knockdowns', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 1, 'name': 'Cuts', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 2, 'name': 'Sig. Strikes to the Head (attempted)', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 3, 'name': 'Sig. Strikes to the Head (landed)', 'bar_flag': 0, 'box_flag': 1},
    {'stat_index': 4, 'name': 'Sig. Strikes to the Body (attempted)', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 5, 'name': 'Sig. Strikes to the Body (landed)', 'bar_flag': 0, 'box_flag': 1},
    {'stat_index': 6, 'name': 'Sig. Strikes to the Legs (attempted)', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 7, 'name': 'Sig. Strikes to the Legs (landed)', 'bar_flag': 0, 'box_flag': 1},
    {'stat_index': 8, 'name': 'Total Sig. Strikes (attempted)', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 9, 'name': 'Total Sig. Strikes (landed)', 'bar_flag': 0, 'box_flag': 1},
    {'stat_index': 10, 'name': 'Sig. Strikes at Distance (attempted)', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 11, 'name': 'Sig. Strikes at Distance (landed)', 'bar_flag': 0, 'box_flag': 1},
    {'stat_index': 12, 'name': 'Sig. Strikes at Clinch (attempted)', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 13, 'name': 'Sig. Strikes at Clinch (landed)', 'bar_flag': 0, 'box_flag': 1},
    {'stat_index': 14, 'name': 'Sig. Strikes at Ground (attempted)', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 15, 'name': 'Sig. Strikes at Ground (landed)', 'bar_flag': 0, 'box_flag': 1},
    {'stat_index': 16, 'name': 'Total Strikes (attempted)', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 17, 'name': 'Total Strikes (landed)', 'bar_flag': 0, 'box_flag': 1},
    {'stat_index': 18, 'name': 'Takedowns (attempted)', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 19, 'name': 'Takedowns (landed)', 'bar_flag': 0, 'box_flag': 1},
    {'stat_index': 20, 'name': 'Submission attempts', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 21, 'name': 'Reversals', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 22, 'name': 'Control time (s)', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 23, 'name': 'Knockdowns - Differential', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 24, 'name': 'Cuts - Differential', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 25, 'name': 'Sig. Strikes to the Head (attempted) - Differential', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 26, 'name': 'Sig. Strikes to the Head - Differential', 'bar_flag': 0, 'box_flag': 1},
    {'stat_index': 27, 'name': 'Sig. Strikes to the Body (attempted) - Differential', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 28, 'name': 'Sig. Strikes to the Body - Differential', 'bar_flag': 0, 'box_flag': 1},
    {'stat_index': 29, 'name': 'Sig. Strikes to the Legs (attempted) - Differential', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 30, 'name': 'Sig. Strikes to the Legs - Differential', 'bar_flag': 0, 'box_flag': 1},
    {'stat_index': 31, 'name': 'Total Sig. Strikes (attempted) - Differential', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 32, 'name': 'Total Sig. Strikes - Differential', 'bar_flag': 0, 'box_flag': 1},
    {'stat_index': 33, 'name': 'Sig. Strikes at Distance (attempted) - Differential', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 34, 'name': 'Sig. Strikes at Distance - Differential', 'bar_flag': 0, 'box_flag': 1},
    {'stat_index': 35, 'name': 'Sig. Strikes at Clinch (attempted) - Differential', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 36, 'name': 'Sig. Strikes at Clinch - Differential', 'bar_flag': 0, 'box_flag': 1},
    {'stat_index': 37, 'name': 'Sig. Strikes at Ground (attempted) - Differential', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 38, 'name': 'Sig. Strikes at Ground - Differential', 'bar_flag': 0, 'box_flag': 1},
    {'stat_index': 39, 'name': 'Total Strikes (attempted) - Differential', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 40, 'name': 'Total Strikes - Differential', 'bar_flag': 0, 'box_flag': 1},
    {'stat_index': 41, 'name': 'Takedowns (attempted) - Differential', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 42, 'name': 'Takedowns - Differential', 'bar_flag': 0, 'box_flag': 1},
    {'stat_index': 43, 'name': 'Submission attempts - Differential', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 44, 'name': 'Reversals - Differential', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': 45, 'name': 'Control time (s) - Differential', 'bar_flag': 1, 'box_flag': 1},
    {'stat_index': -1, 'name': 'Result', 'bar_flag': 1, 'box_flag': 0},
]
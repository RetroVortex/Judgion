# Iván Ontiveros - RetroVortex


# This script encapsulates all the code related to data visualization
# Its purpose is to give information about the dataset to the user
# It can be useful to get an overview of the data, or to get an in-depth understanding of it


# Libraries
from matplotlib import pyplot as plt 
import os
import json
import numpy as np

from judgionLib.utils import stats_getter
from judgionLib.constants import TRAINING_DIRECTORY, STAT_MAP


class DATA_VISUALIZER:

    def __init__(self):

        # This variable will be a matrix with all the fight stats stored
        self.stats = []

        # This variable will be an array with all the round results stored
        self.winners = []

        # This variable will be used for identifying the selected stat (type: int)
        self.stat_selected = 0  # 1: Knockdowns ; 2: Cuts ; 3: Significant strikes attempts (head) ; 4: Significant strikes landed (head)  ; ...

        # This variable will store the name of the selected stat (type: string)
        self.stat_id = ''

        # This variable is used for flow control
        self.keep_going = True

        # These variable are set to True whenever it's required to print the options for the user
        self.options_flag = True
        self.graphs_flag = True

        # Stat maps
        self.bargraph_map = []
        self.boxplot_map = []


    # This function initializes the map variables associated with every type of graph
    def init_maps(self):
        # The '_id' variables are used for setting the value of the "selection_id" keys
        bar_id = 1
        box_id = 1
        
        # Iterate over every stat in the global map
        for stat in STAT_MAP:
            # Bar graph map
            if stat.get('bar_flag') == 1:
                self.bargraph_map.append({
                    'selection_id': bar_id,
                    'name': stat['name'].replace(' (attempted)', ''),   # Removing the 'attempted' part since Bar graphs plot both attempted and landed stats all at once
                    'stat_index': stat['stat_index']
                })
                bar_id += 1
            
            # Box plot map
            if stat.get('box_flag') == 1:
                self.boxplot_map.append({
                    'selection_id': box_id,
                    'name': stat['name'],
                    'stat_index': stat['stat_index']
                })
                box_id += 1


    # This function is used to create the differential vector for a stat
    def compute_differential(self, stat):
        # Extracting the stat values as the differential: (Fighter - Opponent)
        fighter_index = stat - 23
        opponent_index = stat
        # Compute each round once (excluding duplicate rounds)
        v_stat = []
        for round_stats in self.stats[::2]:
            # Stat for red and blue fighter
            fighter_stat = round_stats[fighter_index]
            opponent_stat = round_stats[opponent_index]
            # Store the differential
            differential = fighter_stat - opponent_stat
            v_stat.append(differential)

        return v_stat


    # This is the function that plots graph bars for single stats
    def plot_graphbar_single(self, x, y, ylabel='Rounds'):
        
        # Create the graph
        bars = plt.bar(x, y, color='skyblue', edgecolor='black', linewidth=1, width=0.45)

        # Customize axes
        plt.xlabel(self.stat_id, fontsize=10, loc='center', fontweight='bold', color='gray')
        plt.ylabel(ylabel, fontsize=10, loc='center', fontweight='bold', color='gray')
        plt.title(f'Distribution for {self.stat_id}', fontsize=18, fontweight='bold')

        # Add grid lines
        plt.grid(
            which='both',
            linestyle='--',
            linewidth=0.5,
            color='gray',
            alpha=0.7
        )

        # Add the count number for each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=9,
                color='black'
            )

        # Set limit for y-axis
        plt.ylim(0, max(y) + 1)

        # Layout adjustment prevents clipping
        plt.tight_layout()

        # Show the plot
        plt.show()


    # This is the function that plots graph bars for stats in pairs
    def plot_graphbar_pairs(self, x, cont1, cont2, ylabel='Rounds'):

        bar_width = 0.2

        # Set position of the bars on the x-axis
        r1 = np.arange(len(cont1))
        #r2 = [x + bar_width for x in r1]
        r2 = [r + bar_width for r in r1]

        # Create bars
        plt.bar(r1, cont1, color='skyblue', width=bar_width, edgecolor='grey', label='Attempted')
        plt.bar(r2, cont2, color='blue', width=bar_width, edgecolor='grey', label='Landed')

        # Add xticks on the middle of the group bars
        plt.xlabel(self.stat_id, fontweight='bold', fontsize=10)
        plt.ylabel(ylabel, fontweight='bold', fontsize=10)
        #plt.xticks([r + bar_width/2 for r in range(len(cont1))], x)
        plt.xticks(r1 + bar_width / 2, x)

        # Add grid lines
        plt.grid(
            which='both',
            linestyle='--',
            linewidth=0.5,
            color='gray',
            alpha=0.7
        )

        # Create a legend
        plt.legend()

        # Show the plot
        plt.tight_layout()
        plt.show()


    # This function generates a bar graph for the specified single stat
    def bar_graph(self):
        stat = self.stat_selected

        # If the stat selected is a 'Differential' stat
        if 23 <= stat <= 45:
            v_stat = self.compute_differential(stat)

        # If the stat selected is the 'Result'
        elif stat == -1:
            # Extract the winners (excluding duplicate rounds)
            v_stat = self.winners[::2]

        # Individual stats
        else:
            # Extracting them normally
            v_stat = [fight_stats[self.stat_selected] for fight_stats in self.stats]

        # We plot in one way or another, depending on which stat has been selected
        # First case: [Knockdowns, cuts, submission attempts, reversals]
        if stat in [0, 1, 20, 21]:
            # These stats usually don't go over 3, and are 0 in most cases
            x = ['0', '1', '2', '3', '4(+)']

            # Count values
            cont = [0, 0, 0, 0, 0]
            for i in range(len(v_stat)):
                value = v_stat[i]
                if value == 0:
                    cont[0] += 1
                elif value == 1:
                    cont[1] += 1
                elif value == 2:
                    cont[2] += 1
                elif value == 3:
                    cont[3] += 1
                else:
                    cont[4] += 1
        
            self.plot_graphbar_single(x, cont)

        # Differential stats for the first case
        elif stat in [23, 24, 43, 44]:
            x = ['-3(+)', '-2', '-1', '0', '+1', '+2', '+3(+)']
            cont = [0] * len(x)
            for i in range(len(v_stat)):
                value = v_stat[i]
                if value <= -3:
                    cont[0] += 1
                elif value == -2:
                    cont[1] += 1
                elif value == -1:
                    cont[2] += 1
                elif value == 0:
                    cont[3] += 1
                elif value == 1:
                    cont[4] += 1
                elif value == 2:
                    cont[5] += 1
                else:
                    cont[6] += 1

            self.plot_graphbar_single(x, cont)

        # Second case: Main types of strikes [Head sig. strikes, total sig. strikes, distance sig. strikes, strikes)
        elif stat in [2, 8, 10, 16]:
            # For these values, we plot the 'attempted' (already stored in 'v_stat') and 'landed' (storing it now)
            v_stat2 = [fight_stats[(self.stat_selected + 1)] for fight_stats in self.stats]
            
            # These stats can have a lot of deviation. We are gonna divide them in buckets of size 5
            limits = list(range(0, 101, 5)) + [10000]

            # Setting the labels
            x = [f'{limits[i]}-{limits[i + 1]}' for i in range(len(limits) - 2)] + ['100+']

            #x = [f'{limits[i]}-{limits[i + 1]}' for i in range(len(limits) - 1)] + ['100+']

            # Initialize counters for 'attempted' and 'landed' stats
            cont = [0] * (len(limits) - 1)
            cont2 = [0] * (len(limits) - 1)

            # Count values for the 'attempted' strikes (v_stat)
            for value in v_stat:
                for j in range(len(limits) - 1):
                    if limits[j] <= value < limits[j + 1]:
                        cont[j] += 1
                        break

            # Count values for the 'landed' strikes (v_stat2)
            for value in v_stat2:
                for j in range(len(limits) - 1):
                    if limits[j] <= value < limits[j + 1]:
                        cont2[j] += 1
                        break

            self.plot_graphbar_pairs(x, cont, cont2)

        # Differential stats for the second case
        elif stat in [25, 31, 33, 39]:
            # For these values, we plot the 'attempted' (already stored in 'v_stat') and 'landed' (storing it now)
            v_stat2 = self.compute_differential(stat + 1)
            
            # The differential for these stats has a wide range. 
            limits = [-10000] + list(range(-101, 101, 5)) + [10000]
            
            # Setting the labels
            x = ['-100+'] + [f'{limits[i]}-{limits[i + 1]-1}' for i in range(1, len(limits) - 2)] + ['100+']
            
            # Initialize counters for 'attempted' and 'landed' stats
            cont = [0] * (len(limits) - 1)
            cont2 = [0] * (len(limits) - 1)
            
            # Count values for the 'attempted' strikes (v_stat)
            for value in v_stat:
                for j in range(len(limits) - 1):
                    if limits[j] <= value < limits[j + 1]:
                        cont[j] += 1
                        break
            
            # Count values for the 'landed' strikes (v_stat2)
            for value in v_stat2:
                for j in range(len(limits) - 1):
                    if limits[j] <= value < limits[j + 1]:
                        cont2[j] += 1
                        break
            
            # Plot the paired bar graph
            self.plot_graphbar_pairs(x, cont, cont2)

        # Third case: Secondary types of strikes and Takedowns [body strikes, legs strikes, clinch strikes, ground strikes, takedowns]
        elif stat in [4, 6, 12, 14, 18]:
            # For these values, we plot the 'attempted' (already stored in 'v_stat') and 'landed' (storing it now)
            v_stat2 = [fight_stats[(self.stat_selected + 1)] for fight_stats in self.stats]
            
            # These stats have low values with little deviation for most instances
            limits = [0, 1, 3, 5, 7, 9, 14, 19, 10000]

            # Setting the labels
            x = [
                '0' if i == 0 else (f'{limits[i]}-{limits[i + 1] - 1}' if limits[i + 1] != 10000 else f'{limits[i]}+')
                for i in range(len(limits) - 1)
            ]

            # Initialize counters for 'attempted' and 'landed' stats
            cont = [0] * (len(limits) - 1)
            cont2 = [0] * (len(limits) - 1)

            # Count values for the 'attempted' strikes (v_stat)
            for value in v_stat:
                for j in range(len(limits) - 1):
                    if limits[j] <= value < limits[j + 1]:
                        cont[j] += 1
                        break

            # Count values for the 'landed' strikes (v_stat2)
            for value in v_stat2:
                for j in range(len(limits) - 1):
                    if limits[j] <= value < limits[j + 1]:
                        cont2[j] += 1
                        break

            self.plot_graphbar_pairs(x, cont, cont2)

        # Differential stats for the third case
        elif stat in [27, 29, 35, 37, 41]:
            # For these values, we plot the 'attempted' (already stored in 'v_stat') and 'landed' (storing it now)
            v_stat2 = self.compute_differential(stat + 1)
            
            # These stats range is usually small. 
            limits = [-10000] + list(range(-20, 21, 2)) + [10000]
    
            # Setting the labels
            x = ['-20+'] + [f'{limits[i]}-{limits[i + 1]-1}' for i in range(1, len(limits) - 2)] + ['20+']
            
            # Initialize counters for 'attempted' and 'landed' stats
            cont = [0] * (len(limits) - 1)
            cont2 = [0] * (len(limits) - 1)
            
            # Count values for the 'attempted' strikes (v_stat)
            for value in v_stat:
                for j in range(len(limits) - 1):
                    if limits[j] <= value < limits[j + 1]:
                        cont[j] += 1
                        break
            
            # Count values for the 'landed' strikes (v_stat2)
            for value in v_stat2:
                for j in range(len(limits) - 1):
                    if limits[j] <= value < limits[j + 1]:
                        cont2[j] += 1
                        break
            
            # Plot the paired bar graph
            self.plot_graphbar_pairs(x, cont, cont2)

        # Fourth case: Control time
        elif stat == 22:
            # Unlike the rest of the stats, this one has a max value (300)
            # We do buckets with size 10, while keeping 0 as its own bucket
            limits = [0, 1] + list(range(10, 301, 10))

            # Setting the labels
            x = [
                '0' if i == 0 else (f'{limits[i+1]}')
                for i in range(len(limits) - 1)
            ]

            # Initialize counter and do the count
            cont = [0] * (len(limits) - 1)
            for value in v_stat:
                for j in range(len(limits) - 1):
                    if limits[j] <= value < limits[j + 1]:
                        cont[j] += 1
                        break

            self.plot_graphbar_single(x, cont)

        # Differential control time
        elif stat == 45:
            # Differential can go from -5 minutes to +5
            limits = [-300, -240, -180, -120, -60, 0, 1, 60, 120, 180, 240, 301]
            x = [
                '-300 to -241', '-240 to -181', '-180 to -121', '-120 to -61', '-60 to -1',
                '0',
                '+1 to +59', '+60 to +119', '+120 to +179', '+180 to +239', '+240 to +300'
            ]

            # Initialize the counter array
            cont = [0] * (len(limits) - 1)

            # Count the occurrences in each interval
            for value in v_stat:
                for j in range(len(limits) - 1):
                    if limits[j] <= value < limits[j + 1]:
                        cont[j] += 1
                        break

            self.plot_graphbar_single(x, cont)

        # Fifth case: Result
        else:
            # The result is not a stat of the round, but its label
            # There are only 4 possible outcomes being considered: 10-8 and 10-9 for each fighter
            x = ['10-8 Red', '10-9 Red', '10-9 Blue', '10-8 Blue']

            # Count values
            cont = [0, 0, 0, 0]
            for value in v_stat:
                if value == 0:
                    cont[0] += 1
                elif value == 1:
                    cont[1] += 1
                elif value == 2:
                    cont[2] += 1
                else:
                    cont[3] += 1
        
            self.plot_graphbar_single(x, cont)


    # This function generates a box plot for the specified stat grouped by winner
    def box_plot(self):
        step = 1

        # Differential stats
        if 23 <= self.stat_selected <= 46:
            v_stat = self.compute_differential(self.stat_selected)
            winners = self.winners[::2]
        
        # Normal stats
        else:
            # Extracting the stat values directly
            v_stat = [fight_stats[self.stat_selected] for fight_stats in self.stats]
            winners = self.winners

        # Dictionary where the data will be stored
        data_by_winner = {}
        # 'Result' labels, these will be the keys for the 'data_by_winner' dictionary (the values will be the actual stat values)
        winner_labels = ['10-8 Win', '10-9 Win', '10-9 Loss', '10-8 Loss']

        # Iterate over each round
        for i in range(0, len(winners), step):
            # Extract the result and the number for the stat
            winner = winners[i]
            stat_value = v_stat[i]
            # Storing them in the dictionary
            if winner in data_by_winner:
                data_by_winner[winner].append(stat_value)
            else:
                data_by_winner[winner] = [stat_value]

        # Preprocess data before plotting
        data_to_plot = []
        labels = []
        for winner_code in sorted(data_by_winner.keys()):
            data_to_plot.append(data_by_winner[winner_code])
            labels.append(winner_labels[winner_code])

        # Creating and customizing the box plot
        plt.figure(figsize=(10, 6))
        boxprops = dict(linestyle='-', linewidth=2, color='black')
        medianprops = dict(linestyle='-', linewidth=2, color='gold')
        whiskerprops = dict(linestyle='-', linewidth=1.5, color='gray')
        capprops = dict(linestyle='-', linewidth=2, color='gray')
        flierprops = dict(marker='o', markerfacecolor='red', markersize=5, linestyle='none', markeredgecolor='black')
        plt.boxplot(data_to_plot, labels=labels, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops)
        
        # Titles for axis and graph
        plt.xlabel('Round Outcome', fontsize=10, fontweight='bold')
        plt.ylabel(self.stat_id, fontsize=10, fontweight='bold')
        plt.title(f'{self.stat_id} by Round Outcome', fontsize=14, fontweight='bold')

        # Adding grid lines
        plt.grid(
            which='both',
            linestyle='--',
            linewidth=0.5,
            color='gray',
            alpha=0.7
        )
        
        # Show the plot
        plt.tight_layout()
        plt.show()


    # This function prints which type of graphs are available to plot
    def graph_options(self):
        print("\nAvailable types of graphs to plot:\n")
        print("\t1: Bar Graph")
        print("\t2: Box Plot Grouped by Winner")
        

    # This function prints which stats can be plotted, with their number
    def show_options(self, graph):

        # Options for bar graphs
        if graph == 1:
            print("\nAvailable stats to plot (Bar Graph):\n")
            for stat in self.bargraph_map:
                print(f"\t{stat['selection_id']}: {stat['name']}")
        
        # Options for box plots
        elif graph == 2:
            print("\nAvailable stats to plot:\n")
            for stat in self.boxplot_map:
                print(f"\t{stat['selection_id']}: {stat['name']}")


    # The main function of this class, structuring the whole process
    def visualize_anything(self):  
        
        # Stat maps initialization
        self.init_maps()

        # Stat matrices initialization
        (self.stats, self.winners) = stats_getter(TRAINING_DIRECTORY)
        #print("Stats size: ", len(prueba.stats), ", ", len(self.stats[0]))

        while self.keep_going:
            print("\n------------------------------------------------")

            # Print the graphs options for the user
            if self.graphs_flag:
                self.graph_options()
                self.graphs_flag = False

            try:
                graph_choice = int(input("\nWhich type of graph would you like to plot? Write its number: "))
            except ValueError:
                print("Error: You must write a number.")
                continue

            # Bar Graph
            if graph_choice == 1:
        
                # Print the stats options for the user
                if self.options_flag:
                    self.show_options(1)
                    self.options_flag = False

                # Ask the user for a stat
                try:
                    stat_number = int(input("\nWhich stat would you like to plot? Write its number: "))
                except ValueError:
                    print("Error: You must write a number.")
                    self.options_flag = True
                    continue
                
                # Find the selected stat in the map
                stat_selection = next((item for item in self.bargraph_map if item['selection_id'] == stat_number), None)

                # If found...
                if stat_selection:
                    # Set the variables and plot the bar graph
                    self.stat_selected = stat_selection['stat_index']
                    self.stat_id = stat_selection['name']
                    print("")
                    self.bar_graph()

                    # Ask the user if they want to keep plotting
                    c = input("Do you want to plot another value? [Y/N]: ")
                    if c.lower() == 'n':
                        self.keep_going = False

                # If not found... (invalid stat selected)
                else:
                    # Repeat the process showing the available options again for the user
                    self.options_flag = True
                    print("The number selected does not belong to any stat in the list.")
                    continue

            # Box Plot Grouped by Winner
            elif graph_choice == 2:

                # Print the stats options for the user
                if self.options_flag:
                    self.show_options(2)
                    self.options_flag = False

                # Ask the user for a stat
                try:
                    stat_number = int(input("\nWhich stat would you like to plot? Write its number: "))
                except ValueError:
                    print("Error: You must write a number")
                    self.options_flag = True
                    continue

                # Find the selected stat in boxplot_map
                stat_selection = next((item for item in self.boxplot_map if item['selection_id'] == stat_number), None)

                # If found...
                if stat_selection:

                    # Set the variables and plot the box plot grouped by winner
                    self.stat_selected = stat_selection['stat_index']
                    self.stat_id = stat_selection['name']
                    print("")
                    self.box_plot()

                    # Ask the user if they want to keep plotting
                    c = input("Do you want to plot another value? [Y/N]: ")
                    if c.lower() == 'n':
                        self.keep_going = False

                # If not found... (invalid stat selected)
                else:
                    # Repeat the process showing the available options again for the user
                    self.options_flag = True
                    print("The number selected does not belong to any stat in the list.")
                    continue

            # Invalid graph selected
            else:
                self.graphs_flag = True
                print("The number selected does not belong to any graph option in the list.")
                continue



# Main
if __name__ == '__main__':
    prueba = DATA_VISUALIZER()
    prueba.visualize_anything()


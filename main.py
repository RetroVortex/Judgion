# Iván Ontiveros - RetroVortex


# This script executes the AI-powered judging process for a fight 
# Initializing 3 judges who will decide the winner, just as in real life


# Libraries
from round_judge import AI_JUDGE
from json_generator import UFC_WEB_SCRAPER
from judgionLib.constants import NUM_JUDGES


# This function is used to declare the winner based on the judges' scorecards
def declare_winner(results, red_name, blue_name):

    # Counting the wins
    red_wins = results.count(1)
    blue_wins = results.count(2)
    draw_votes = results.count(0)

    # Printing the final result
    print("\n-------------------")
    print("| OFFICIAL DECISION |")
    print("-------------------\n")

    # Processing the winner. This method does not assume there are 3 scorecards.
    if draw_votes == 0:
        if blue_wins == 0:
            print(f"Winner by Unanimous Decision: {red_name}")
        elif red_wins == 0:
            print(f"Winner by Unanimous Decision: {blue_name}")
        else:
            if red_wins > blue_wins:
                print(f"Winner by Split Decision: {red_name}")
            elif blue_wins > red_wins:
                print(f"Winner by Split Decision: {blue_name}")
            else:
                print("Split Draw")
    
    else:
        if draw_votes == NUM_JUDGES:
            print("Unanimous Draw")

        else:

            if red_wins > 0 and blue_wins > 0:
                if draw_votes >= red_wins: 
                    if draw_votes >= blue_wins:
                        print("Split Draw")
                    else:
                        print(f"Winner by Split Decision: {blue_name}")
                elif draw_votes >= blue_wins:
                    if draw_votes >= red_wins:
                        print("Split Draw")
                    else:
                        print(f"Winner by Split Decision: {red_name}")

            elif red_wins == 0:
                if draw_votes > blue_wins:
                    print("Majority Draw")
                else:
                    print(f"Winner by Majority Decision: {blue_name}")

            else:
                if draw_votes > red_wins:
                    print("Majority Draw")
                else:
                    print(f"Winner by Majority Decision: {red_name}")



if __name__ == '__main__':

    # Init the judges
    judges = []
    for i in range(NUM_JUDGES):
        judges.append(AI_JUDGE())
        judges[i].set_judge()

    # Ask the user for the link and init the web scraper
    fight_link = input("Enter the fight link (UFC Stats website link): ")
    scraper = UFC_WEB_SCRAPER(fight_link, False, True)      # We generate the files in the testing directory, not the training directory

    # Do the scraping
    scraper.start_scraping()

    # Store the fight ID from the scraper
    identificator = f"{scraper.red_last_name}_{scraper.blue_last_name}"

    # The user can change the file now (useful for adding cuts or changing the label)
    input("\nYou can change the generated JSON files if needed (f.e. the 'cuts' stat).\nPress Enter when all is set to start the judging process.")

    # Request the scorecards
    results = []
    for judge in judges:

        # First, set the fight ID so the model can open the requested files
        judge.fight_id = identificator

        # Then, do the judging process and store the result --> 0: Draw ; 1: Red fighter win ; 2: Blue fighter win
        results.append(judge.give_scorecards())

    # Process the winner 
    declare_winner(results, scraper.red_last_name, scraper.blue_last_name)





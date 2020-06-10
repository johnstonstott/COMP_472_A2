# -------------------------------------------------------
# Assignment 2
# Written by Johnston Stott (40059176)
# For COMP 472 Section ABIX – Summer 2020
# --------------------------------------------------------

# Permitted libs: Numpy, Pandas, sklearn, Nltk, Matplotlib, math and sys

# Row | Object ID | Title | Post Type | Author | Created At | URL | Points | Number of comments | year
# 0   | 1         | 2     | 3         | 4      | 5          | 6   | 7      | 8                  | 9

import csv
import os
import sys

import functions
import word

# Task 1: Extract data and build model.

# vocabulary is a list of Word objects for title words, used for creating the model.
vocabulary = []
# testing_set is a list of Word objects for title words, used for testing.
testing_set = []

csv_file = "hns_2018_2019.csv"

if not os.path.exists(csv_file):
    print("File ", csv_file, "not found. Please make sure it is in the same directory as main.py")
    print("Program terminating.")
    sys.exit(0)

with open(csv_file, "r") as file:
    print("Opening", csv_file, "and reading data... ", end="")
    reader = csv.reader(file)

    # Iterate through data set and add to the appropriate list depending on the year.
    for entry in reader:
        if entry[5][0:4] == "2018":
            title_words = functions.clean_words(entry[2])
            post_type = entry[3]
            functions.add_to_vocabulary(title_words, post_type, vocabulary)
        elif entry[5][0:4] == "2019":
            title_words = functions.clean_words(entry[2])
            post_type = entry[3]
            functions.add_to_vocabulary(title_words, post_type, testing_set)

    print("Done\n")

# vocab_size is number of unique words.
vocab_size = len(vocabulary)

# These are how many words are in each post category.
nb_of_story = functions.count_story_posts(vocabulary)
nb_of_ask_hn = functions.count_ask_hn_posts(vocabulary)
nb_of_show_hn = functions.count_show_hn_posts(vocabulary)
nb_of_poll = functions.count_poll_posts(vocabulary)

# Create model-2018.txt file.
model_file = "model-2018.txt"
if os.path.exists(model_file):
    print("The file", model_file, "already exists, so not creating a new one.")
else:
    print("Creating model file... ", end="")

    # TODO sort?
    count = 0

    with open(model_file, "w") as file:
        for v in vocabulary:
            count += 1
            word = v.content
            freq_story = str(v.freq_story)
            prob_story = str((freq_story + 0.5) / (nb_of_story + vocab_size * 0.5))
            freq_ask_hn = str(v.freq_ask_hn)
            prob_ask_hn = str((freq_ask_hn + 0.5) / (nb_of_ask_hn + vocab_size * 0.5))
            freq_show_hn = str(v.freq_show_hn)
            prob_show_hn = str((freq_show_hn + 0.5) / (nb_of_show_hn + vocab_size * 0.5))
            freq_poll = str(v.freq_poll)
            prob_poll = str((freq_poll + 0.5) / (nb_of_poll + vocab_size * 0.5))
            file.write(str(count) + "  " + word + "  " + freq_story + "  " + prob_story + "  " + freq_ask_hn + "  " +
                       prob_ask_hn + "  " + freq_show_hn + "  " + prob_show_hn + "  " + freq_poll + "  " + prob_poll +
                       "\n")

            # count, "  ", word, "  ", freq_story, "  ", prob_story, "  ", freq_ask_hn, "  ",
            #                  prob_ask_hn, "  ", freq_show_hn, "  ", prob_show_hn, "  ", freq_poll, "  ", prob_poll)

            # count, "  ", word + "  " + freq_story + "  " + prob_story + "  " + freq_ask_hn + "  " + prob_ask_hn + "  "
            # + freq_show_hn + "  " + prob_show_hn + "  " + freq_poll + "  " + prob_poll + "\n"


    print("Done")
print("Model file can be found in model-2018.txt\n")

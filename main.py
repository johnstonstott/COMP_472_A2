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

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import functions
import word

# x = ["freq = 1", "freq ≤ 5", "freq ≤ 10", "freq ≤ 15", "freq ≤ 20"]
# y_1 = [60, 70, 80, 90, 100]
# y_2 = [50, 55, 60, 65, 70]
# y_3 = [90, 95, 80, 85, 100]
# y_4 = [10, 20, 30, 40, 70]
# plt.plot(x, y_1, marker="o", label="one")
# plt.plot(x, y_2, marker="o", label="two")
# plt.plot(x, y_3, marker="o", label="three")
# plt.plot(x, y_4, marker="o", label="four")
# plt.legend(loc="upper left")
# plt.xlabel("X's")
# plt.ylabel("Y's")
# plt.show()

# Task 1: Extract data and build model.
print("STARTING TASK 1\n")

# vocabulary is a list of Word objects for title words, used for creating the model.
vocabulary = []
# testing_set is a dictionary of <title>: <post type>, used for testing.
testing_set = {}

# Stores the count of how many titles are in each type.
count_story_titles = 0
count_ask_hn_titles = 0
count_show_hn_titles = 0
count_poll_titles = 0

csv_file = "hns_2018_2019.csv"

if not os.path.exists(csv_file):
    print("File ", csv_file, "not found. Please make sure it is in the same directory as main.py")
    print("Program terminating.")
    sys.exit(0)

with open(csv_file, "r") as file:
    print(f"Opening {csv_file}, reading data, creating vocabulary, creating testing set... ", end="")
    reader = csv.reader(file)

    # Iterate through data set and add to the appropriate list depending on the year.
    for entry in reader:
        if entry[5][0:4] == "2018":
            title_words = functions.clean_words(entry[2])
            post_type = entry[3]
            functions.add_to_vocabulary(title_words, post_type, vocabulary)

            # Add to count of how many posts for each type.
            if post_type == "story":
                count_story_titles += 1
            elif post_type == "ask_hn":
                count_ask_hn_titles += 1
            elif post_type == "show_hn":
                count_show_hn_titles += 1
            elif post_type == "poll":
                count_poll_titles += 1

        elif entry[5][0:4] == "2019":
            title_string = functions.clean_title(entry[2])
            post_type = entry[3]
            testing_set.update([(title_string.strip(), post_type)])

    print("Done\n")

# Create a sorted vocabulary for making the output files, sort by word alphabetically.
vocabulary_sorted = sorted(vocabulary, key=lambda x: x.content)

# Create model-2018.txt file.
model_file = "model-2018.txt"
functions.create_model_file(model_file, vocabulary_sorted)

# Create vocabulary.txt file.
vocab_file = "vocabulary.txt"
functions.create_vocabulary_file(vocab_file, vocabulary_sorted)

print("TASK 1 COMPLETE\n")

# Task 2: Use ML classifier to test dataset.
print("STARTING TASK 2\n")

# Create baseline-result.txt file.
result_file = "baseline-result.txt"
functions.create_result_file(result_file, model_file, testing_set, count_story_titles, count_ask_hn_titles,
                             count_show_hn_titles, count_poll_titles)

print("TASK 2 COMPLETE\n")

# Task 3: Experiments with the classifier.
# Task 3.1: Stop-word filtering.
print("STARTING TASK 3.1\n")

# Filter stopwords from complete vocabulary and store them in a new vocabulary.
stopwords_file = "stopwords.txt"
vocabulary_31 = functions.filter_vocabulary_by_stopword(vocabulary_sorted, stopwords_file)

# Create the model file from the vocabulary.
model_file_31 = "stopword-model.txt"
functions.create_model_file(model_file_31, vocabulary_31)

# Use ML classifier to test dataset.
result_file_31 = "stopword-result.txt"
functions.create_result_file(result_file_31, model_file_31, testing_set, count_story_titles, count_ask_hn_titles,
                             count_show_hn_titles, count_poll_titles)

print("TASK 3.1 COMPLETE\n")

# Task 3.2: Word length filtering.
print("STARTING TASK 3.2\n")

# Get Word objects from vocabulary fitting the criteria of task 3.2.
min_length = 2
max_length = 9
vocabulary_32 = functions.filter_vocabulary_by_length(vocabulary_sorted, min_length, max_length)

# Create the model file.
model_file_32 = "wordlength-model.txt"
functions.create_model_file(model_file_32, vocabulary_32)

# Classify data from testing set.
result_file_32 = "wordlength-result.txt"
functions.create_result_file(result_file_32, model_file_32, testing_set, count_story_titles, count_ask_hn_titles,
                             count_show_hn_titles, count_poll_titles)

print("TASK 3.2 COMPLETE")

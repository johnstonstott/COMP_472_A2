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
if os.path.exists(vocab_file):
    print("The file", vocab_file, "already exists, so not creating a new one")
else:
    print("Creating vocabulary file... ", end="")

    with open(vocab_file, "w") as file:
        for v in vocabulary_sorted:
            file.write(v.content + "\n")

    print("Done")
print("Vocabulary file can be found in vocabulary.txt\n")

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

# Stores Word objects that fit the criteria of task 3.1.
vocabulary_31 = []

stopwords_file = "stopwords.txt"
stopwords = functions.extract_stopwords(stopwords_file)

# Creating new vocabulary using the one defined in task 1.
print("Reading existing vocabulary, removing stop words, creating new vocabulary... ", end="")

# Check if each one is equivalent to any stopwords, if so then don't include it in the new vocabulary.
for i in range(len(vocabulary_sorted)):
    if vocabulary_sorted[i].content not in stopwords:
        vocabulary_31.append(vocabulary_sorted[i])

print("Done\n")

model_file_31 = "stopword-model.txt"
functions.create_model_file(model_file_31, vocabulary_31)

# Use ML classifier to test dataset.
result_file_31 = "stopword-result.txt"
functions.create_result_file(result_file_31, model_file_31, testing_set, count_story_titles, count_ask_hn_titles,
                             count_show_hn_titles, count_poll_titles)

print("TASK 3.1 COMPLETE\n")

# Task 3.2: Word length filtering.
print("STARTING TASK 3.2\n")

# Stores Word objects from vocabulary fitting the criteria of task 3.2.
vocabulary_32 = []
min_length = 2
max_length = 9

print("Reading existing vocabulary, removing words outside of length limits, creating new vocabulary... ", end="")

# Check word lengths and only add ones within the accepted range.
for i in range(len(vocabulary_sorted)):
    if min_length < len(vocabulary_sorted[i].content) < max_length:
        vocabulary_32.append(vocabulary_sorted[i])

print("Done\n")

# Create the model file.
model_file_32 = "wordlength-model.txt"
functions.create_model_file(model_file_32, vocabulary_32)

# Classify data from testing set.
result_file_32 = "wordlength-result.txt"
functions.create_result_file(result_file_32, model_file_32, testing_set, count_story_titles, count_ask_hn_titles,
                             count_show_hn_titles, count_poll_titles)

print("TASK 3.2 COMPLETE")

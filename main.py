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

# count_total_titles is the total number of posts in the training set
count_total_titles = count_story_titles + count_ask_hn_titles + count_show_hn_titles + count_poll_titles

# vocab_size is number of unique words.
vocab_size = len(vocabulary)

# These are how many words are in each post category, used for calculating probability in model.
count_story_words = functions.count_story_posts(vocabulary)
count_ask_hn_words = functions.count_ask_hn_posts(vocabulary)
count_show_hn_words = functions.count_show_hn_posts(vocabulary)
count_poll_words = functions.count_poll_posts(vocabulary)

# Create a sorted vocabulary for making the output files, sort by word alphabetically.
vocabulary_sorted = sorted(vocabulary, key=lambda x: x.content)

# Create model-2018.txt file.
model_file = "model-2018.txt"
if os.path.exists(model_file):
    print("The file", model_file, "already exists, so not creating a new one")
else:
    print("Creating model file... ", end="")
    count = 0

    # Probabilities with 0.5 smoothing are calculated with the formula:
    # probability_of_wi = (count_of_wi + 0.5) / (number_of_words_in_post_type + (vocabulary_size * 0.5))
    with open(model_file, "w") as file:
        for v in vocabulary_sorted:
            count += 1
            count_str = str(count)
            word = v.content
            freq_story = v.freq_story
            prob_story = (freq_story + 0.5) / (count_story_words + vocab_size * 0.5)
            freq_story = str(freq_story)
            prob_story = str(prob_story)
            freq_ask_hn = v.freq_ask_hn
            prob_ask_hn = (freq_ask_hn + 0.5) / (count_ask_hn_words + vocab_size * 0.5)
            freq_ask_hn = str(freq_ask_hn)
            prob_ask_hn = str(prob_ask_hn)
            freq_show_hn = v.freq_show_hn
            prob_show_hn = (freq_show_hn + 0.5) / (count_show_hn_words + vocab_size * 0.5)
            freq_show_hn = str(freq_show_hn)
            prob_show_hn = str(prob_show_hn)
            freq_poll = v.freq_poll
            prob_poll = (freq_poll + 0.5) / (count_poll_words + vocab_size * 0.5)
            freq_poll = str(freq_poll)
            prob_poll = str(prob_poll)

            file.write(count_str + "  " + word + "  " + freq_story + "  " + prob_story + "  " + freq_ask_hn + "  "
                       + prob_ask_hn + "  " + freq_show_hn + "  " + prob_show_hn + "  " + freq_poll + "  " + prob_poll
                       + "\n")

    print("Done")
print("Model file can be found in model-2018.txt\n")

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
baseline_file = "baseline-result.txt"
if os.path.exists(baseline_file):
    print("The file", baseline_file, "already exists, so not creating a new one")
else:
    print("Classifying testing set and creating baseline result file... ", end="")
    count = 0
    model_list = functions.extract_model_data(model_file)

    with open(baseline_file, "w") as file:
        for key in testing_set:
            count += 1
            post_title = key

            # Compute the probability of each post type.
            score_story = functions.compute_score(key, model_list, "story", count_story_titles, count_total_titles)
            score_ask_hn = functions.compute_score(key, model_list, "ask_hn", count_ask_hn_titles, count_total_titles)
            score_show_hn = functions.compute_score(key, model_list, "show_hn", count_show_hn_titles,
                                                    count_total_titles)
            score_poll = functions.compute_score(key, model_list, "poll", count_poll_titles, count_total_titles)

            # Determine which probability is the highest.
            score_max = max(score_story, score_ask_hn, score_show_hn, score_poll)
            predicted_type = ""

            if score_max == score_story:
                predicted_type = "story"
            elif score_max == score_ask_hn:
                predicted_type = "ask_hn"
            elif score_max == score_show_hn:
                predicted_type = "show_hn"
            elif score_max == score_poll:
                predicted_type = "poll"

            # Compare and see if the prediction was correct.
            actual_type = testing_set[key]
            prediction_result = ""

            if actual_type == predicted_type:
                prediction_result = "right"
            else:
                prediction_result = "wrong"

            # Use all the information above and write to the file.
            count_str = str(count)
            score_story = str(score_story)
            score_ask_hn = str(score_ask_hn)
            score_show_hn = str(score_show_hn)
            score_poll = str(score_poll)

            file.write(count_str + "  " + post_title + "  " + predicted_type + "  " + score_story + "  "
                       + score_ask_hn + "  " + score_show_hn + "  " + score_poll + "  " + actual_type + "  "
                       + prediction_result + "\n")

    print("Done")

print("Baseline result file can be found in baseline-result.txt\n")

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

# How many words are in filtered vocabulary.
vocab_size_31 = len(vocabulary_31)

# These are how many words are in each post category, used for calculating probability in model.
count_story_words = functions.count_story_posts(vocabulary_31)
count_ask_hn_words = functions.count_ask_hn_posts(vocabulary_31)
count_show_hn_words = functions.count_show_hn_posts(vocabulary_31)
count_poll_words = functions.count_poll_posts(vocabulary_31)

model_file_31 = "stopword-model.txt"
if os.path.exists(model_file_31):
    print("The file", model_file_31, "already exists, so not creating a new one")
else:
    print("Creating model file... ", end="")
    count = 0

    # Probabilities with 0.5 smoothing are calculated with the formula:
    # probability_of_wi = (count_of_wi + 0.5) / (number_of_words_in_post_type + (vocabulary_size * 0.5))
    with open(model_file_31, "w") as file:
        for v in vocabulary_31:
            count += 1
            count_str = str(count)
            word = v.content
            freq_story = v.freq_story
            prob_story = (freq_story + 0.5) / (count_story_words + vocab_size_31 * 0.5)
            freq_story = str(freq_story)
            prob_story = str(prob_story)
            freq_ask_hn = v.freq_ask_hn
            prob_ask_hn = (freq_ask_hn + 0.5) / (count_ask_hn_words + vocab_size_31 * 0.5)
            freq_ask_hn = str(freq_ask_hn)
            prob_ask_hn = str(prob_ask_hn)
            freq_show_hn = v.freq_show_hn
            prob_show_hn = (freq_show_hn + 0.5) / (count_show_hn_words + vocab_size_31 * 0.5)
            freq_show_hn = str(freq_show_hn)
            prob_show_hn = str(prob_show_hn)
            freq_poll = v.freq_poll
            prob_poll = (freq_poll + 0.5) / (count_poll_words + vocab_size_31 * 0.5)
            freq_poll = str(freq_poll)
            prob_poll = str(prob_poll)

            file.write(count_str + "  " + word + "  " + freq_story + "  " + prob_story + "  " + freq_ask_hn + "  "
                       + prob_ask_hn + "  " + freq_show_hn + "  " + prob_show_hn + "  " + freq_poll + "  " + prob_poll
                       + "\n")

    print("Done")
print("Model file can be found in", model_file_31, "\n")

# Use ML classifier to test dataset.
baseline_file_31 = "stopword-result.txt"

if os.path.exists(baseline_file_31):
    print("The file", baseline_file_31, "already exists, so not creating a new one")
else:
    print("Classifying testing set and creating stopword result file... ", end="")
    count = 0
    model_list = functions.extract_model_data(model_file_31)

    with open(baseline_file_31, "w") as file:
        for key in testing_set:
            count += 1
            post_title = key

            # Compute the probability of each post type.
            score_story = functions.compute_score(key, model_list, "story", count_story_titles, count_total_titles)
            score_ask_hn = functions.compute_score(key, model_list, "ask_hn", count_ask_hn_titles, count_total_titles)
            score_show_hn = functions.compute_score(key, model_list, "show_hn", count_show_hn_titles,
                                                    count_total_titles)
            score_poll = functions.compute_score(key, model_list, "poll", count_poll_titles, count_total_titles)

            # Determine which post type has the highest probability.
            score_max = max(score_story, score_ask_hn, score_show_hn, score_poll)
            predicted_type = ""

            if score_max == score_story:
                predicted_type = "story"
            elif score_max == score_ask_hn:
                predicted_type = "ask_hn"
            elif score_max == score_show_hn:
                predicted_type = "show_hn"
            elif score_max == score_poll:
                predicted_type = "poll"

            # Compare and see if the prediction was correct.
            actual_type = testing_set[key]
            prediction_result = ""

            if actual_type == predicted_type:
                prediction_result = "right"
            else:
                prediction_result = "wrong"

            # Write to the file.
            count_str = str(count)
            score_story = str(score_story)
            score_ask_hn = str(score_ask_hn)
            score_show_hn = str(score_show_hn)
            score_poll = str(score_poll)

            file.write(count_str + "  " + post_title + "  " + predicted_type + "  " + score_story + "  "
                       + score_ask_hn + "  " + score_show_hn + "  " + score_poll + "  " + actual_type + "  "
                       + prediction_result + "\n")

    print("Done")
print("Stopword result file can be found in", baseline_file_31, "\n")

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

print("Done")

# Count of words in filtered vocabulary.
vocab_size_32 = len(vocabulary_32)

# Number of words per category for use in probability calculations.
count_story_words = functions.count_story_posts(vocabulary_32)
count_ask_hn_words = functions.count_ask_hn_posts(vocabulary_32)
count_show_hn_words = functions.count_show_hn_posts(vocabulary_32)
count_poll_words = functions.count_poll_posts(vocabulary_32)

# Create the model file.
model_file_32 = "wordlength-model.txt"
if os.path.exists(model_file_32):
    print("The file", model_file_32, "already exists, so not creating a new one")
else:
    print("Creating model file... ", end="")
    count = 0

    # Probabilities with 0.5 smoothing are calculated with the formula:
    # probability_of_wi = (count_of_wi + 0.5) / (number_of_words_in_post_type + (vocabulary_size * 0.5))
    with open(model_file_32, "w") as file:
        for v in vocabulary_32:
            count += 1
            count_str = str(count)
            word = v.content
            freq_story = v.freq_story
            prob_story = (freq_story + 0.5) / (count_story_words + vocab_size_32 * 0.5)
            freq_story = str(freq_story)
            prob_story = str(prob_story)
            freq_ask_hn = v.freq_ask_hn
            prob_ask_hn = (freq_ask_hn + 0.5) / (count_ask_hn_words + vocab_size_32 * 0.5)
            freq_ask_hn = str(freq_ask_hn)
            prob_ask_hn = str(prob_ask_hn)
            freq_show_hn = v.freq_show_hn
            prob_show_hn = (freq_show_hn + 0.5) / (count_show_hn_words + vocab_size_32 * 0.5)
            freq_show_hn = str(freq_show_hn)
            prob_show_hn = str(prob_show_hn)
            freq_poll = v.freq_poll
            prob_poll = (freq_poll + 0.5) / (count_poll_words + vocab_size_32 * 0.5)
            freq_poll = str(freq_poll)
            prob_poll = str(prob_poll)

            file.write(count_str + "  " + word + "  " + freq_story + "  " + prob_story + "  " + freq_ask_hn + "  "
                       + prob_ask_hn + "  " + freq_show_hn + "  " + prob_show_hn + "  " + freq_poll + "  " + prob_poll
                       + "\n")

    print("Done")
print("Model file can be found in", model_file_32, "\n")

# Classify data from testing set.
baseline_file_32 = "wordlength-result.txt"

if os.path.exists(baseline_file_32):
    print("The file", baseline_file_32, "already exists, so not creating a new one")
else:
    print("Classifying testing set and creating wordlength result file... ", end="")
    count = 0
    model_list = functions.extract_model_data(model_file_32)

    with open(baseline_file_32, "w") as file:
        for key in testing_set:
            count += 1
            post_title = key

            # Compute probability of each post type.
            score_story = functions.compute_score(key, model_list, "story", count_story_titles, count_total_titles)
            score_ask_hn = functions.compute_score(key, model_list, "ask_hn", count_ask_hn_titles, count_total_titles)
            score_show_hn = functions.compute_score(key, model_list, "show_hn", count_show_hn_titles,
                                                    count_total_titles)
            score_poll = functions.compute_score(key, model_list, "poll", count_poll_titles, count_total_titles)

            # Determine which score is highest.
            score_max = max(score_story, score_ask_hn, score_show_hn, score_poll)
            predicted_type = ""

            if score_max == score_story:
                predicted_type = "story"
            elif score_max == score_ask_hn:
                predicted_type = "ask_hn"
            elif score_max == score_show_hn:
                predicted_type = "show_hn"
            elif score_max == score_poll:
                predicted_type = "poll"

            # Compare and see if prediction was correct.
            actual_type = testing_set[key]
            prediction_result = ""

            if actual_type == predicted_type:
                prediction_result = "right"
            else:
                prediction_result = "wrong"

            # Write to file.
            count_str = str(count)
            score_story = str(score_story)
            score_ask_hn = str(score_ask_hn)
            score_show_hn = str(score_show_hn)
            score_poll = str(score_poll)

            file.write(count_str + "  " + post_title + "  " + predicted_type + "  " + score_story + "  "
                       + score_ask_hn + "  " + score_show_hn + "  " + score_poll + "  " + actual_type + "  "
                       + prediction_result + "\n")

    print("Done")
print("Wordlength result file can be found in", baseline_file_32, "\n")

print("TASK 3.2 COMPLETE")

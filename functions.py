# -------------------------------------------------------
# Assignment 2
# Written by Johnston Stott (40059176)
# For COMP 472 Section ABIX – Summer 2020
# --------------------------------------------------------

import math
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import word


# Remove unwanted characters from beginning and end of string and then add them to a list.
def clean_words(title):
    words = title.split()
    remove_word = ["(", ")", "[", "]", "{", "}", "<", ">", "«", "»", " ", "‍‍‍", "   ", ":", ".", "?", "!", "&", "=",
                   "@", "'", "‘", "’", "`", '"', "“", "”", ",", "–", "—", "-", "^", "/", "\\", "", "\n", "\t"]
    clean = []

    for w in words:
        string = w
        word_clean = False

        if string in remove_word:
            continue

        while word_clean is False:
            if len(string) == 0:
                break

            if string[0] in remove_word:
                string = string[1:]

            if len(string) > 0 and string[-1] in remove_word:
                string = string[0:-1]

            if len(string) > 0 and string[0] not in remove_word and string[-1] not in remove_word:
                word_clean = True

        if len(string) == 0:
            continue

        clean.append(string.lower())

    return clean


# Remove unwanted characters and return the title as a single string with all unwanted characters removed.
def clean_title(title):
    words = title.split()
    remove_word = ["(", ")", "[", "]", "{", "}", "<", ">", "«", "»", " ", "‍‍‍", "   ", ":", ".", "?", "!", "&", "=",
                   "@", "'", "‘", "’", "`", '"', "“", "”", ",", "–", "—", "-", "^", "/", "\\", "", "\n", "\t"]
    clean = ""

    for w in words:
        string = w
        word_clean = False

        if string in remove_word:
            continue

        while word_clean is False:
            if len(string) == 0:
                break

            if string[0] in remove_word:
                string = string[1:]

            if len(string) > 0 and string[-1] in remove_word:
                string = string[0:-1]

            if len(string) > 0 and string[0] not in remove_word and string[-1] not in remove_word:
                word_clean = True

        if len(string) == 0:
            continue

        clean += string.lower() + " "

    return clean


# Goes through vocabulary to check if a word has already been added, returns the index or -1.
def get_word_index(new_word, vocab):
    for i in range(len(vocab)):
        if vocab[i].content == new_word:
            return i
    return -1


# Check through list when calculating score to find its index, returns the index or -1.
def get_model_index(title_word, model_list):
    for i in range(len(model_list)):
        if model_list[i][1] == title_word:
            return i
    return -1


# Adds words to the vocabulary list while taking into account the post type.
def add_to_vocabulary(words, post_type, vocab):
    for w in words:
        index = get_word_index(w, vocab)
        # Word exists, increment its frequency.
        if index != -1:
            vocab[index].increment_count(post_type)
        # Word does not exist, add it.
        else:
            new_word = word.Word(w)
            new_word.increment_count(post_type)
            vocab.append(new_word)


# Takes a file of stopwords and an existing vocabulary, returns a new vocabulary with the stopwords removed.
def filter_vocabulary_by_stopword(vocab, stopwords_file):
    print("Reading existing vocabulary, removing stop words, creating new vocabulary... ", end="")
    stopwords = extract_stopwords(stopwords_file)
    new_vocab = []

    # Check if each one is equivalent to any stopwords, if so then don't include it in the new vocabulary.
    for i in range(len(vocab)):
        if vocab[i].content not in stopwords:
            new_vocab.append(vocab[i])

    print("Done\n")
    return new_vocab


# Takes a min and max length and existing vocabulary, returns a new vocabulary containing only words within that length.
def filter_vocabulary_by_length(vocab, min_len, max_len):
    print("Reading existing vocabulary, removing words outside of length limits, creating new vocabulary... ", end="")
    new_vocab = []

    # Check lengths of each word and only add ones within the accepted range.
    for i in range(len(vocab)):
        if min_len < len(vocab[i].content) < max_len:
            new_vocab.append(vocab[i])

    print("Done\n")
    return new_vocab


# Take an existing vocabulary and return a new vocabulary with all words removed below the frequency limit.
def filter_vocabulary_by_frequency(vocab, freq_limit):
    print(f"Reading existing vocabulary, removing words with frequency ≤ {freq_limit}... ", end="")
    new_vocab = []

    for i in range(len(vocab)):
        if vocab[i].freq_total > freq_limit:
            new_vocab.append(vocab[i])

    print("Done")
    return new_vocab


# Filters existing vocabulary by removing words above with frequency above a certain percent.
def filter_vocabulary_by_frequency_percent(orig_vocab, trimmed_vocab, freq_percent, freq_list):
    print(f"Reading existing vocabulary, removing the top {int(freq_percent * 100)}% of words... ", end="")

    # Determine what frequency is the maximum allowed.
    max_freq = freq_list[math.floor(freq_percent * len(orig_vocab)) - 1]
    new_vocab = []

    for i in range(len(trimmed_vocab)):
        if trimmed_vocab[i].freq_total < max_freq:
            new_vocab.append(trimmed_vocab[i])

    print("Done")
    return new_vocab


# Goes through list and determines total number of words in story posts.
def count_story_posts(words):
    total = 0

    for w in words:
        freq = w.freq_story
        total += freq

    return total


# Goes through list and determines total number of words in ask_hn posts.
def count_ask_hn_posts(words):
    total = 0

    for w in words:
        freq = w.freq_ask_hn
        total += freq

    return total


# Goes through list and determines total number of words in show_hn posts.
def count_show_hn_posts(words):
    total = 0

    for w in words:
        freq = w.freq_show_hn
        total += freq

    return total


# Goes through list and determines total number of words in poll posts.
def count_poll_posts(words):
    total = 0

    for w in words:
        freq = w.freq_poll
        total += freq

    return total


# Returns a sorted list of all frequencies in the vocabulary which will be used to determine the top 5%, 10%, etc.
def get_frequency_list(vocab):
    frequencies = []

    for v in vocab:
        frequencies.append(v.freq_total)

    frequencies.sort(reverse=True)
    return frequencies


# Calculates probabilities, creates model, and writes it to a file.
def create_model_file(file_name, vocab):
    vocab_size = len(vocab)

    # These are how many words are in each post category, used for calculating probability.
    count_story_words = count_story_posts(vocab)
    count_ask_hn_words = count_ask_hn_posts(vocab)
    count_show_hn_words = count_show_hn_posts(vocab)
    count_poll_words = count_poll_posts(vocab)

    if os.path.exists(file_name):
        print("The file", file_name, "already exists, so not creating a new one, delete or rename", file_name, "to "
              "create a new one")
    else:
        print("Creating model file... ", end="")
        count = 0

        # Conditional probabilities with 0.5 smoothing are calculated with the formula:
        # probability_of_wi = (count_of_wi + 0.5) / (number_of_words_in_post_type + (vocabulary_size * 0.5))
        with open(file_name, "w") as file:
            for v in vocab:
                count += 1
                count_str = str(count)
                word_content = v.content
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

                file.write(count_str + "  " + word_content + "  " + freq_story + "  " + prob_story + "  " + freq_ask_hn
                           + "  " + prob_ask_hn + "  " + freq_show_hn + "  " + prob_show_hn + "  " + freq_poll + "  "
                           + prob_poll + "\n")

        print("Done")
    print("Model file can be found in", file_name, "\n")


# Does the same as the above function but returns a 2D list of the information instead of writing to a file.
def create_model_list(vocab):
    print("Creating model... ", end="")
    vocab_size = len(vocab)
    count = 0
    model_list = []

    # These are how many words are in each post category, used for calculating probability.
    count_story_words = count_story_posts(vocab)
    count_ask_hn_words = count_ask_hn_posts(vocab)
    count_show_hn_words = count_show_hn_posts(vocab)
    count_poll_words = count_poll_posts(vocab)

    # Conditional probabilities with 0.5 smoothing are calculated with the formula:
    # probability_of_wi = (count_of_wi + 0.5) / (number_of_words_in_post_type + (vocabulary_size * 0.5))
    for v in vocab:
        count += 1
        word_content = v.content
        freq_story = v.freq_story
        prob_story = (freq_story + 0.5) / (count_story_words + vocab_size * 0.5)
        freq_ask_hn = v.freq_ask_hn
        prob_ask_hn = (freq_ask_hn + 0.5) / (count_ask_hn_words + vocab_size * 0.5)
        freq_show_hn = v.freq_show_hn
        prob_show_hn = (freq_show_hn + 0.5) / (count_show_hn_words + vocab_size * 0.5)
        freq_poll = v.freq_poll
        prob_poll = (freq_poll + 0.5) / (count_poll_words + vocab_size * 0.5)
        word_list = [count, word_content, freq_story, prob_story, freq_ask_hn, prob_ask_hn, freq_show_hn, prob_show_hn,
                     freq_poll, prob_poll]
        model_list.append(word_list)

    print("Done")
    return model_list


# Write each word of a vocabulary in a file, each on a new line.
def create_vocabulary_file(file_name, vocab):
    if os.path.exists(file_name):
        print("The file", file_name, "already exists, so not creating a new one, delete or rename", file_name, "to "
              "create a new one")
    else:
        print("Creating vocabulary file... ", end="")

        with open(file_name, "w") as file:
            for v in vocab:
                file.write(v.content + "\n")

        print("Done")
    print("Vocabulary file can be found in", file_name, "\n")


# Read model file, iterate through testing set and predict the post type for each in an output file.
def create_result_file(file_name, model_file, test_set, count_sty, count_ask, count_shw, count_pol):
    if os.path.exists(file_name):
        print("The file", file_name, "already exists, so not creating a new one, delete or rename", file_name, "to "
              "create a new one")
    else:
        print("Classifying testing set and creating result file... ", end="")
        count = 0
        count_tot = count_sty + count_ask + count_shw + count_pol
        model_list = extract_model_data(model_file)

        with open(file_name, "w") as file:
            for key in test_set:
                count += 1
                post_title = key

                # Compute the probability of each post type.
                score_story = compute_score(key, model_list, "story", count_sty, count_tot)
                score_ask_hn = compute_score(key, model_list, "ask_hn", count_ask, count_tot)
                score_show_hn = compute_score(key, model_list, "show_hn", count_shw, count_tot)
                score_poll = compute_score(key, model_list, "poll", count_pol, count_tot)

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
                actual_type = test_set[key]

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
    print("Result file can be found in", file_name, "\n")


# Takes a list with all model info, test set, and counts of post types, returns a list with the predicted and actual
# results for classifications.
def create_result_lists(model_list, test_set, count_sty, count_ask, count_shw, count_pol):
    print("Classifying testing set... ", end="")
    count_tot = count_sty + count_ask + count_shw + count_pol
    predicted_results = []
    actual_results = []

    for key in test_set:
        # Compute the probability of each post type.
        score_story = compute_score(key, model_list, "story", count_sty, count_tot)
        score_ask_hn = compute_score(key, model_list, "ask_hn", count_ask, count_tot)
        score_show_hn = compute_score(key, model_list, "show_hn", count_shw, count_tot)
        score_poll = compute_score(key, model_list, "poll", count_pol, count_tot)

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

        # Add the predicted post type and actual post type to the same position in their respective lists.
        predicted_results.append(predicted_type)
        actual_results.append(test_set[key])

    # Return both lists.
    print("Done")
    return [predicted_results, actual_results]


# Reads the model file and produces a 2D list with all of its data.
def extract_model_data(model):
    model_list = []

    # Read data from model file.
    with open(model, "r") as file:
        contents = file.readlines()

    # Add each line into model_list.
    for c in contents:
        model_list.append(c.split())

    return model_list


# Creates and returns a list of all words in the stopwords file.
def extract_stopwords(stopwords):
    stopwords_list = []

    # Read and append words to list.
    with open(stopwords, "r") as file:
        contents = file.readlines()

    for c in contents:
        stopwords_list.append(c.strip())

    return stopwords_list


# Uses the model to determine the probability that a title belongs to a certain post type.
def compute_score(title, model_data, post_type, type_count, total_count):
    title_words = title.split()

    # Score is calculated with the formula:
    # score(post_type) = log(P(post_type)) + log(P(w_1 | post_type)) + ... + log(P(w_n | post_type))

    # This is the part of the equation:
    # log(P(post_type))
    score = 0.0

    # Avoid division by 0 error in log10 if there are no instances of a certain post type.
    # Return the lowest possible probability.
    if type_count == 0:
        return float(-math.inf)

    prob_type = np.log10(type_count / total_count)
    score += prob_type

    # This is the part of the equation:
    # log(P(w_1 | post_type)) + ... + log(P(w_n | post_type))
    for i in range(len(title_words)):
        word_index = get_model_index(title_words[i], model_data)

        # Word is not in our model, so ignore it.
        if word_index == -1:
            continue

        prob_word_given_type = 0

        if post_type == "story":
            prob_word_given_type = model_data[word_index][3]
        elif post_type == "ask_hn":
            prob_word_given_type = model_data[word_index][5]
        elif post_type == "show_hn":
            prob_word_given_type = model_data[word_index][7]
        elif post_type == "poll":
            prob_word_given_type = model_data[word_index][9]

        prob_word_given_type = float(prob_word_given_type)
        score += np.log10(prob_word_given_type)

    return score


# Takes lists to store the results, and a list containing the actual/predicted, calculates results and stores them.
def compute_metrics(result_list, acc_list, rcl_list, prs_list, fms_list):
    print("Calculating metrics... ", end="")
    predicted_list = result_list[0]
    actual_list = result_list[1]

    # Calculate accuracy.
    acc_list.append(accuracy_score(predicted_list, actual_list))

    # Calculate recall.
    rcl_list.append(recall_score(predicted_list, actual_list, average="macro"))

    # Calculate precision.
    prs_list.append(precision_score(predicted_list, actual_list, average="macro"))

    # Calculate f-measure.
    fms_list.append(f1_score(predicted_list, actual_list, average="macro"))

    print("Done\n")


# Use matplotlib to show the graph of the four performance indicators.
def create_and_show_graph(x_vals, y_vals_acc, y_vals_rcl, y_vals_prc, y_vals_fms, title):
    print("When you are done viewing the graph, please close it to continue")

    # Accuracy, recall, precision, f-measure are 4 distinct lines on the graph.
    plt.plot(x_vals, y_vals_acc, marker="o", label="Accuracy")
    plt.plot(x_vals, y_vals_rcl, marker="o", label="Recall")
    plt.plot(x_vals, y_vals_prc, marker="o", label="Precision")
    plt.plot(x_vals, y_vals_fms, marker="o", label="F-Measure")
    plt.title(title)
    plt.xlabel("Words in vocabulary")
    plt.ylabel("Performance")
    plt.legend(loc="best")

    print("Displaying results... ", end="")
    plt.show()
    print("Done\n")

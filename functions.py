# -------------------------------------------------------
# Assignment 2
# Written by Johnston Stott (40059176)
# For COMP 472 Section ABIX – Summer 2020
# --------------------------------------------------------

import math
import os

import numpy as np

import word


# Remove unwanted characters from beginning and end of string and then add them to a list.
def clean_words(title):
    words = title.split()
    remove_word = ["(", ")", "[", "]", "{", "}", "<", ">", "«", "»", " ", "‍‍‍", "   ", ":", ".", "?", "!", "&", "=",
                   "@",
                   "'", "‘", "’", "`", '"', "“", "”", ",", "–", "—", "-", "^", "/", "\\", "", "\n", "\t"]
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
                   "@",
                   "'", "‘", "’", "`", '"', "“", "”", ",", "–", "—", "-", "^", "/", "\\", "", "\n", "\t"]
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
        if index != -1:
            vocab[index].increment_count(post_type)
        else:
            new_word = word.Word(w)
            new_word.increment_count(post_type)
            vocab.append(new_word)


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
        return
    else:
        print("Creating model file... ", end="")
        count = 0

        # Probabilities with 0.5 smoothing are calculated with the formula:
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

        print("Done\nModel file can be found in", file_name, "\n")


# Read model file, iterate through testing set and predict the post type for each in an output file.
def create_result_file(file_name, model_file, test_set, count_sty, count_ask, count_shw, count_pol):
    if os.path.exists(file_name):
        print("The file", file_name, "already exists, so not creating a new one, delete or rename", file_name, "to "
              "create a new one")
        return
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

        print("Done\nResult file can be found in", file_name, "\n")


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

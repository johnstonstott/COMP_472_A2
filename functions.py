# -------------------------------------------------------
# Assignment 2
# Written by Johnston Stott (40059176)
# For COMP 472 Section ABIX – Summer 2020
# --------------------------------------------------------

import numpy as np

import word


# Remove unwanted characters from beginning and end of string and then add them to a list.
def clean_words(title):
    words = title.split()
    remove_word = ["(", ")", "[", "]", "{", "}", "<", ">", "«", "»", " ", "‍‍‍", "   ", ":", ".", "?", "!", "&", "=", "@",
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
    remove_word = ["(", ")", "[", "]", "{", "}", "<", ">", "«", "»", " ", "‍‍‍", "   ", ":", ".", "?", "!", "&", "=", "@",
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


# Uses the model to determine the probability that a title belongs to a certain post type.
def compute_score(title, model, post_type, type_count, total_count):
    title_words = title.split()
    model_list = []

    # Read data from model file.
    with open(model, "r") as file:
        contents = file.readlines()

    # Add each line into model_list.
    for c in contents:
        model_list.append(c.split())

    # Score is calculated with the formula:
    # score(post_type) = log(P(post_type)) + log(P(w_1 | post_type)) + ... + log(P(w_n | post_type))

    # This is the part of the equation:
    # log(P(post_type))
    score = 0.0
    prob_type = np.log10(type_count / total_count)
    score += prob_type

    # This is the part of the equation:
    # log(P(w_1 | post_type)) + ... + log(P(w_n | post_type))
    for i in range(len(title_words)):
        word_index = get_model_index(title_words[i], model_list)

        # Word is not in our model, so ignore it.
        if word_index == -1:
            continue

        prob_word_given_type = 0

        if post_type == "story":
            prob_word_given_type = model_list[word_index][3]
        elif post_type == "ask_hn":
            prob_word_given_type = model_list[word_index][5]
        elif post_type == "show_hn":
            prob_word_given_type = model_list[word_index][7]
        elif post_type == "poll":
            prob_word_given_type = model_list[word_index][9]

        prob_word_given_type = float(prob_word_given_type)
        score += np.log10(prob_word_given_type)

    return score

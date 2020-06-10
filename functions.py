# -------------------------------------------------------
# Assignment 2
# Written by Johnston Stott (40059176)
# For COMP 472 Section ABIX – Summer 2020
# --------------------------------------------------------

import word


# Remove unwanted characters from beginning and end of string and then add them to a list.
def clean_words(title):
    words = title.split()
    remove_word = ["(", ")", "[", "]", "{", "}", "<", ">", ":", ".", "?", "!", "'", "’", ",", "“", "”", "–"]
    clean = []

    for w in words:
        string = w

        if string in remove_word:
            continue

        if string[0] in remove_word:
            string = string[1:]

        if string[-1] in remove_word:
            string = string[0:-1]

        if len(string) == 0:
            continue

        clean.append(string.lower())

    return clean


# Goes through vocabulary to check if a word has already been added, returns the index or -1.
def get_word_index(new_word, vocab):
    for i in range(len(vocab)):
        if vocab[i].content == new_word:
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

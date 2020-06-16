# -------------------------------------------------------
# Assignment 2
# Written by Johnston Stott (40059176)
# For COMP 472 Section ABIX – Summer 2020
# --------------------------------------------------------


class Word:
    def __init__(self, content):
        self.content = content
        self.freq_story = 0
        self.freq_ask_hn = 0
        self.freq_show_hn = 0
        self.freq_poll = 0
        self.freq_total = 0

    def __str__(self):
        return f"{self.content} - {self.freq_story} {self.freq_ask_hn} {self.freq_show_hn} {self.freq_poll} - " \
               f"{self.freq_total}"

    # Increments the correct frequency counter based on the post type.
    def increment_count(self, post_type):
        if post_type == "story":
            self.freq_story += 1
        elif post_type == "ask_hn":
            self.freq_ask_hn += 1
        elif post_type == "show_hn":
            self.freq_show_hn += 1
        elif post_type == "poll":
            self.freq_poll += 1

        self.freq_total += 1

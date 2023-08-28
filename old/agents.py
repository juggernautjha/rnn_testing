# [markdown]
# # Trexquant Interview Project (The Hangman Game)
# 
# * Copyright Trexquant Investment LP. All Rights Reserved. 
# * Redistribution of this question without written consent from Trexquant is prohibited

# ## Instruction:
# For this coding test, your mission is to write an algorithm that plays the game of Hangman through our API server. 
# 
# When a user plays Hangman, the server first selects a secret word at random from a list. The server then returns a row of underscores (space separated)—one for each letter in the secret word—and asks the user to guess a letter. If the user guesses a letter that is in the word, the word is redisplayed with all instances of that letter shown in the correct positions, along with any letters correctly guessed on previous turns. If the letter does not appear in the word, the user is charged with an incorrect guess. The user keeps guessing letters until either (1) the user has correctly guessed all the letters in the word
# or (2) the user has made six incorrect guesses.
# 
# You are required to write a "guess" self.get_histogramtion that takes current word (with underscores) as input and returns a guess letter. You will use the API codes below to play 1,000 Hangman games. You have the opportunity to practice before you want to start recording your game results.
# 
# Your algorithm is permitted to use a training set of approximately 250,000 dictionary words. Your algorithm will be tested on an entirely disjoint set of 250,000 dictionary words. Please note that this means the words that you will ultimately be tested on do NOT appear in the dictionary that you are given. You are not permitted to use any dictionary other than the training dictionary we provided. This requirement will be strictly enforced by code review.
# 
# You are provided with a basic, working algorithm. This algorithm will match the provided masked string (e.g. a _ _ l e) to all possible words in the dictionary, tabulate the frequency of letters appearing in these possible words, and then guess the letter with the highest frequency of appearence that has not already been guessed. If there are no remaining words that match then it will default back to the character frequency distribution of the entire dictionary.
# 
# This benchmark strategy is successful approximately 18% of the time. Your task is to design an algorithm that significantly outperforms this benchmark.#
import json
import requests
import random
import string
import secrets
import time
import re
import collections
from hangman import Hangman
try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode


import pandas as pd

from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

#! Barebones Python implementation of Hangman, fuelled by 2 bottles of Sting and your mom.
import numpy as np
from typing import List, Tuple, Dict, Union
# import core
# requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


class HangmanAPI(object):
    #! API Provided by Trexquant, I am not making any changes here other than fixing self.hangman_url for testing purposes.
    def __init__(self, access_token=None, session=None, timeout=None, wordlist="data/250k_train.txt"):
        self.hangman_url = 'https://trexsim.com/trexsim/hangman'
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []
        full_dictionary_location = wordlist
        self.full_dictionary = self.build_dictionary(full_dictionary_location)      
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        
        self.current_dictionary = []
        

    
    @staticmethod
    def determine_hangman_url():
        links = ['https://trexsim.com', 'https://sg.trexsim.com']

        data = {link: 0 for link in links}

        for link in links:

            requests.get(link)

            for i in range(10):
                s = time.time()
                requests.get(link)
                data[link] = time.time() - s

        link = sorted(data.items(), key=lambda x: x[1])[0][0]
        link += '/trexsim/hangman'
        return link

    def guess(self, word): # word input example: "_ p p _ e "
        ###############################################
        # Replace with your own "guess" self.get_histogramtion here #
        ###############################################

        # clean the word so that we strip away the space characters
        # replace "_" with "." as "." indicates any character in regular expressions
        clean_word = word[::2].replace("_",".")
        
        # find length of passed word
        len_word = len(clean_word)
        
        # grab current dictionary of possible words from self object, initialize new possible words dictionary to empty
        current_dictionary = self.current_dictionary
        new_dictionary = []
        
        # iterate through all of the words in the old plausible dictionary
        for dict_word in current_dictionary:
            # continue if the word is not of the appropriate length
            if len(dict_word) != len_word:
                continue
                
            # if dictionary word is a possible match then add it to the current dictionary
            if re.match(clean_word,dict_word):
                new_dictionary.append(dict_word)
        
        # overwrite old possible words dictionary with updated version
        self.current_dictionary = new_dictionary
        
        
        # count occurrence of all characters in possible word matches
        full_dict_string = "".join(new_dictionary)
        
        c = collections.Counter(full_dict_string)
        sorted_letter_count = c.most_common()                   
        
        guess_letter = '!'
        
        # return most frequently occurring letter in all possible words that hasn't been guessed yet
        for letter,instance_count in sorted_letter_count:
            if letter not in self.guessed_letters:
                guess_letter = letter
                break
            
        # if no word matches in training dictionary, default back to ordering of full dictionary
        if guess_letter == '!':
            sorted_letter_count = self.full_dictionary_common_letter_sorted
            for letter,instance_count in sorted_letter_count:
                if letter not in self.guessed_letters:
                    guess_letter = letter
                    break            
        
        return guess_letter

    ##########################################################
    # You'll likely not need to modify any of the code below #
    ##########################################################
    
    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location,"r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary
                
    def start_game(self, practice=True, verbose=True):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary #! The misleading line. 
                         
        response = self.request("/new_game", {"practice":practice})
        if response.get('status')=="approved":
            game_id = response.get('game_id')
            word = response.get('word')
            tries_remains = response.get('tries_remains')
            if verbose:
                print("Successfully start a new game! Game ID: {0}. # of tries remaining: {1}. Word: {2}.".format(game_id, tries_remains, word))
            while tries_remains>0:
                # get guessed letter from user code
                guess_letter = self.guess(word)
                    
                # append guessed letter to guessed letters field in hangman object
                self.guessed_letters.append(guess_letter)
                if verbose:
                    print("Guessing letter: {0}".format(guess_letter))
                    
                try:    
                    res = self.request("/guess_letter", {"request":"guess_letter", "game_id":game_id, "letter":guess_letter})
                except HangmanAPIError:
                    print('HangmanAPIError exception caught on request.')
                    continue
                except Exception as e:
                    print('Other exception caught on request.')
                    raise e
               
                if verbose:
                    print("Sever response: {0}".format(res))
                status = res.get('status')
                tries_remains = res.get('tries_remains')
                if status=="success":
                    if verbose:
                        print("Successfully finished game: {0}".format(game_id))
                    return True
                elif status=="failed":
                    reason = res.get('reason', '# of tries exceeded!')
                    if verbose:
                        print("Failed game: {0}. Because of: {1}".format(game_id, reason))
                    return False
                elif status=="ongoing":
                    word = res.get('word')
        else:
            if verbose:
                print("Failed to start a new game")
        return status=="success"
        
    def my_status(self):
        return self.request("/my_status", {})
    
    def request(
            self, path, args=None, post_args=None, method=None):
        if args is None:
            args = dict()
        if post_args is not None:
            method = "POST"

        # Add `access_token` to post_args or args if it has not already been
        # included.
        if self.access_token:
            # If post_args exists, we assume that args either does not exists
            # or it does not need `access_token`.
            if post_args and "access_token" not in post_args:
                post_args["access_token"] = self.access_token
            elif "access_token" not in args:
                args["access_token"] = self.access_token

        time.sleep(0.2)

        num_retry, time_sleep = 50, 2
        for it in range(num_retry):
            try:
                response = self.session.request(
                    method or "GET",
                    self.hangman_url + path,
                    timeout=self.timeout,
                    params=args,
                    data=post_args,
                    verify=False
                )
                break
            except requests.HTTPError as e:
                response = json.loads(e.read())
                raise HangmanAPIError(response)
            except requests.exceptions.SSLError as e:
                if it + 1 == num_retry:
                    raise
                time.sleep(time_sleep)

        headers = response.headers
        if 'json' in headers['content-type']:
            result = response.json()
        elif "access_token" in parse_qs(response.text):
            query_str = parse_qs(response.text)
            if "access_token" in query_str:
                result = {"access_token": query_str["access_token"][0]}
                if "expires" in query_str:
                    result["expires"] = query_str["expires"][0]
            else:
                raise HangmanAPIError(response.json())
        else:
            raise HangmanAPIError('Maintype was not text, or querystring')

        if result and isinstance(result, dict) and result.get("error"):
            raise HangmanAPIError(result)
        return result
    
class HangmanAPIError(Exception):
    def __init__(self, result):
        self.result = result
        self.code = None
        try:
            self.type = result["error_code"]
        except (KeyError, TypeError):
            self.type = ""

        try:
            self.message = result["error_description"]
        except (KeyError, TypeError):
            try:
                self.message = result["error"]["message"]
                self.code = result["error"].get("code")
                if not self.type:
                    self.type = result["error"].get("type", "")
            except (KeyError, TypeError):
                try:
                    self.message = result["error_msg"]
                except (KeyError, TypeError):
                    self.message = result

        Exception.__init__(self, self.message)


# self.get_histogramtion to find number of times a letter come in whole dictionary, keeping count of letter 1 if it comes in a word else 0
# def func(new_dictionary):
#     dictx = collections.Counter()
#     for words in new_dictionary:
#         # print(words)
#         temp = collections.Counter(words)
#         # print(temp)
#         for i in temp:
#             temp[i] = 1
#         dictx = dictx + temp
#     return dictx



def get_histogram(dictionary):
    res = {chr(i) : 0 for i in range(ord('a'), ord('z')+1)}
    for wrd in dictionary:
        for ltr in set(wrd):
            res[ltr] += 1
    return collections.Counter(res)

# def func2(n_word_dictionary, clean_word):
#     new_dictionary = []
#     l = len(clean_word)
#     for dict_word in n_word_dictionary[l]:
#         if re.match(clean_word,dict_word):
#             new_dictionary.append(dict_word)
#     return new_dictionary


def isolate_substrings(n_word_dictionary, pattern, length):
    new_dictionary = list(filter(pattern.match, n_word_dictionary[length]))
    return new_dictionary

def isolate_from_pattern(n_word_dictionary, pattern):
    length = len(pattern)
    pattern = re.compile(pattern)
    new_dictionary = list(filter(pattern.match, n_word_dictionary[length]))
    return new_dictionary

class ngramSolver(HangmanAPI):
    def __init__(self, access_token=None, session=None, wordlist = None,timeout=None):
        super().__init__(access_token, session, timeout)
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []
        self.max_length = 0
        for words in self.full_dictionary:
            if(len(words)>self.max_length):
                self.max_length = len(words)
        full_dictionary_location = "data/250k.txt"
        self.full_dictionary = self.build_dictionary(full_dictionary_location)        
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        
        self.current_dictionary = []
        self.vowels = ['a', 'e', 'i', 'o', 'u']

        #! Generate n-grams from 3 onwards.
        self.n_word_dictionary = {i:[] for i in range(3, 30)}
        count = 3
        while count<=self.max_length:
            for words in self.full_dictionary:
                if(len(words)>=count):
                    for i in range(len(words)-count+1):
                        self.n_word_dictionary[count].append(words[i:i+count]) 
            count = count+1
        
    
    def vowel_count(self, wrd):
        count = 0
        for i in wrd:
            if i in self.vowels: count = count+1.0
        return count/len(wrd)


    def get_histogram_flattened(self, dictionary):
        #! Approach 1 : Count every occurrence just once, as in Rahuul contributes just one u to the count.
        #! RESULT : Poor accuracy because test and train datasets are disjoint and it is unwise to expect word_wise_letter_counts to be same.
        res = {chr(i) : 0 for i in range(ord('a'), ord('z')+1)}
        for wrd in dictionary:
            for ltr in set(wrd):
                res[ltr] += 1
        res = {i : res[i] for i in res if res[i] != 0}
        return collections.Counter(res)
    


    def get_histogram(self, dictionary):
        #! Gets plain letter frequency. Definitely works better than the flattened_histogram approach.
        histogram = collections.Counter()
        for word in dictionary  :
            temp = collections.Counter(word)
            for i in temp:
                temp[i] = 1
                histogram = histogram + temp
        return histogram
        
    def guess(self, word): # word input example: "_ p p _ e "
        clean_word = word[::2].replace("_",".")
        len_word = len(clean_word)
        
        current_dictionary = self.current_dictionary
        pattern = re.compile(f'\\b{clean_word}\\b')  #! same size, and same pattern.
        patt = re.compile(clean_word)
        my_dic = list(filter(pattern.match, current_dictionary))
        self.current_dictionary = my_dic        
        c = self.get_histogram(self.current_dictionary)
        letter_frequencies = c.most_common()                   
        guess_letter = '!'
        
        # return most frequently occurring letter in all possible words that hasn't been guessed yet
        for letter, _ in letter_frequencies:
            if letter not in self.guessed_letters:
                if letter in self.vowels and self.vowel_count(clean_word)>0.55:
                    #! If more than 55% of the word is vowels, do not guess any more vowels.
                    self.guessed_letters.append(letter)
                    continue
                guess_letter = letter
                break
                
        #! This executes only when current_dictionary is empty.
        if guess_letter == '!':
            new_dictionary = isolate_substrings(self.n_word_dictionary, patt, len_word)
            c = self.get_histogram(new_dictionary)
            sorted_letter_count = c.most_common()
            for letter,_ in sorted_letter_count:
                if letter not in self.guessed_letters:
                    if letter in self.vowels and self.vowel_count(clean_word)>0.55:
                        self.guessed_letters.append(letter)
                        continue
                    guess_letter = letter
                    break
                    
        #! Larger n-gram. Not limiting to (say, 5 gram) because that gave poor results.            
        if guess_letter == '!':
            x = int(len(clean_word)/2)
            if x>=3:
                c = collections.Counter()
                for i in range(len(clean_word)-x+1):
                    s = clean_word[i:i+x]
                    new_dictionary = isolate_from_pattern(self.n_word_dictionary, s)
                    temp = self.get_histogram(new_dictionary)
                    c = c+temp
                sorted_letter_count = c.most_common()
                for letter, instance_count in sorted_letter_count:
                    if letter not in self.guessed_letters:
                        guess_letter = letter
                        break
                    
        #! Finer n-gram. Use when larger n-grams do not work.            
        if guess_letter == '!':
            x = int(len(clean_word)/3)
            if x>=3:
                c = collections.Counter()
                for i in range(len(clean_word)-x+1):
                    s = clean_word[i:i+x]
                    new_dictionary = isolate_from_pattern(self.n_word_dictionary, s)
                    temp = self.get_histogram(new_dictionary)
                    c = c+temp
                sorted_letter_count = c.most_common()
                for letter, _ in sorted_letter_count:
                    if letter not in self.guessed_letters:
                        guess_letter = letter
                        break
            
            
        #! Fallback: if no word matches in training dictionary, default back to ordering of full dictionary
        if guess_letter == '!':
            sorted_letter_count = self.full_dictionary_common_letter_sorted
            for letter,_ in sorted_letter_count:
                if letter not in self.guessed_letters:
                    if letter in self.vowels and self.vowel_count(clean_word)>0.55:
                        self.guessed_letters.append(letter)
                        continue
                    guess_letter = letter
                    break            
        
        return guess_letter

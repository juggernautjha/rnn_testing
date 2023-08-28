from agents import HangmanAPI
from hangman import Hangman
import collections
import re
import random


global THRESHOLD
#! Local version of the hangmanAPI for testing. Made changes to the 
class LocalHangman(HangmanAPI):
    #! Class defined by me for testing locally because HangmanAPI has limited number of training games, and it was rate limited.
    def __init__(self, access_token=None, session=None, timeout=None, wordlist = "data/250k_train.txt"):
        super().__init__(access_token, session, timeout)
        self.full_dictionary = self.build_dictionary(wordlist)
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        self.current_dictionary =[]
        self.call_sign = "local_default" #! For logging purposes.
        
    def guess(self, word): # word input example: "_ p p _ e "
        ###############################################
        # Replace with your own "guess" function here #
        ###############################################

        # clean the word so that we strip away the space characters
        # replace "_" with "." as "." indicates any character in regular expressions
        # print(word)
        clean_word = word.replace("_",".") #! only modification.
        # print(clean_word)
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
    
    def start_game(self, practice: bool =True, verbose: bool =True, host_dict : str = "250k_complement.txt"):
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary
        game = Hangman(f"data/{host_dict}",6)
        word = game.state
        tries_remains = game.max_tries - game.curr_tries
        game_id = game.game_id
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
                res = game.evaluate_guess(guess_letter)
            except Exception as e:
                print('Other exception caught on request.')
                raise e
           
            if verbose:
                print("Sever response: {0}".format(res))
            status = res['status']
            tries_remains = res['tries_remains']
            if status=="success":
                if verbose:
                    print("Successfully finished game: {0}".format(game_id))
                    print(f"\n===============================================================================\n")
                return True
            elif status=="failed":
                if verbose:
                    print("Failed game: {0}. Because of: {1}".format(game_id, "tries exceeded"))
                    print(f"\n===============================================================================\n")
                return False
            elif status=="ongoing":
                word = res['word']

def isolate_substrings(n_word_dictionary, pattern, length):
    new_dictionary = []
    try:
        new_dictionary = list(filter(pattern.match, n_word_dictionary[length]))
    except:
        pass
    return new_dictionary

def isolate_from_pattern(n_word_dictionary, pattern):
    new_dictionary = []
    length = len(pattern)
    pattern = re.compile(pattern)
    try:
        new_dictionary = list(filter(pattern.match, n_word_dictionary[length]))
    except:
        pass
    return new_dictionary

class localGreedy(LocalHangman):
    def guess(self, word): # word input example: "_ p p _ e "
    ###############################################
    # Replace with your own "guess" function here #
    ###############################################

    # clean the word so that we strip away the space characters
    # replace "_" with "." as "." indicates any character in regular expressions
    # print(word)
        clean_word = word.replace("_",".") #! only modification.
        # print(clean_word)
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
                trial = random.randint(1,10)
                if trial < 5:
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




class ngramSolver(LocalHangman):
    def __init__(self, access_token=None, session=None, wordlist = None,timeout=None, threshold = 55):
        super().__init__(access_token, session, timeout)
        self.guessed_letters = []
        self.max_length = 0
        for words in self.full_dictionary:
            if(len(words)>self.max_length):
                self.max_length = len(words)
        full_dictionary_location = "data/250k.txt"
        self.full_dictionary = self.build_dictionary(full_dictionary_location)        
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        self.call_sign = "ngram_fiend"
        self.current_dictionary = []
        self.vowels = ['a', 'e', 'i', 'o', 'u']
        self.threshold = threshold

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
        clean_word = word.replace("_",".")
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
                if letter in self.vowels and self.vowel_count(clean_word)>self.threshold:
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
                    if letter in self.vowels and self.vowel_count(clean_word)>self.threshold:
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
                    if letter in self.vowels and self.vowel_count(clean_word)>self.threshold:
                        self.guessed_letters.append(letter)
                        continue
                    guess_letter = letter
                    break            
        
        return guess_letter
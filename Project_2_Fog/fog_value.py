import re
import string
import os
import math
import english_syllable
import csv
from nltk.corpus import cmudict
phoneme_dict = dict(cmudict.entries())


def remove_html_tags(line):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', line)


def is_disclosure(line):
    return line[:len("Disclosure:")] == "Disclosure:"


def not_empty(line):
    word = re.compile(r'\w')
    if re.findall(word, line):
        return True
    else:
        return False


def delete_initial_spaces(line):
    delete_init_spaces = re.compile(r'  +(\w.+)')
    return ''.join(re.findall(delete_init_spaces, line))


# count number of sentences in one line
def extract_sentences(line):
    # Start with a Capital letter and end with a lower-case letter followed by {. ? !}
    sentence = re.compile(r'[A-Z].+?[a-z][\.\?\!]')
    # findall returns non-overlapping matches,
    # this function can deal with the following special cases:
    # Word like J.C. does not matter because the final character should be lower-case.
    # Word like vs. does not matter because although it cuts off a sentence, the real full stop won't be counted again
    # unless an abbreviated word like Inc. (Capital initial letter and full stop in one word) occurs.
    return re.findall(sentence, line)


# count number of words in one line
def extract_words(line):
    char = re.compile(r'[A-Za-z/-]')
    words = line.split(" ")  # words are separated by spaces
    clean_words = []
    for word in words:
        # extract all English characters including hyphen in the word for syllabus counting
        clean_word = "".join(re.findall(char, word))
        if clean_word:
            clean_words.append(clean_word)
    return clean_words


def preprocess(file):
    n_sents = 0
    n_words = 0
    all_words = dict()

    with open(file, 'r', encoding='UTF-8-sig') as inp:
        data = inp.readlines()
    for line in data:
        clean_line = remove_html_tags(line)
        clean_line = delete_initial_spaces(clean_line)
        if is_disclosure(clean_line):  # Skip the Disclosure section
            continue
        sents = extract_sentences(clean_line)
        words = extract_words(clean_line)
        n_sents += len(sents)
        n_words += len(words)
        if words:
            for word in words:
                if word not in all_words.keys():
                    all_words[word] = 1
                else:
                    all_words[word] += 1

    assert sum(all_words.values()) == n_words

    return {
        "n_sents": n_sents,
        "n_words": n_words,
        "words_dict": all_words
    }


def syllables_in_word(word):
    """
    Count number of syllables in a word
    First use cmu dictionary (use phoneme) this would be fairly accurate.
    If the word cannot be found in the dictionary, use pyphen's hyphenation algorithm, which has flaws
    e.g. it predicts rhythm as one syllable

    """
    word = word.lower()
    if word in phoneme_dict.keys():
        # return sum([ phoneme.count(str(num)) for phoneme in phoneme_dict[word] for num in range(3) ])
        return len([ph for ph in phoneme_dict[word] if ph.strip(string.ascii_letters)])
    else:
        n_sys = english_syllable.count(word)
        return n_sys


def is_complex(word, filter_list=[]):
    if word in filter_list:
        return False    # Filter out common financial words
    elif len(word) >= 15:
        return True  # Long word must contain at least 3 syllables
    else:
        return syllables_in_word(word) > 2  # Complex words are defined as words with more than 2 syllables


# load all file data from the given directory
def load_all(dir_path):
    lst_of_dicts = []
    os.chdir(dir_path)
    for file in os.listdir(os.curdir):
        w_dict = preprocess(file)['words_dict']
        lst_of_dicts.append(w_dict)
    assert len(lst_of_dicts) == len(os.listdir(os.curdir))
    return lst_of_dicts


def load_filter_list():
    with open('complex.txt', 'r', encoding='UTF-8-sig') as inp:
        return [word.replace('\n', '') for word in inp.readlines()]


def weight_of_word(word, lst_of_dicts):
    N = len(lst_of_dicts)
    df = 0
    for dic in lst_of_dicts:
        if word in dic.keys():
            df += 1
    return math.log(N / df) / math.log(N)


# Three types of fogs, the original one, the weighted one, the weighted one with the filter list
def fog(file, filter_list, lst_of_dicts, mode='o'):
    """
    :param file: input article
    :param mode: {o, w, w+l} i.e. the original one, the weighted one, the weighted one with the filter list
    :param filter_list: common financial words list
    :return: fog value of the input article
    """
    fl = filter_list if mode == 'w+l' else []
    # Count necessary stats
    stat = preprocess(file)
    n_sents = stat['n_sents']
    n_words = stat['n_words']
    dic = stat['words_dict']
    complex_words = [word for word in dic if is_complex(word, fl)]
    words_per_sentence = n_words / n_sents
    if mode == 'o':
        percent_of_complex_words = 100 * sum(dic[cw] for cw in complex_words) / n_words  # the original complex percentage
    elif mode == 'w' or 'w+l':
        percent_of_complex_words = 100 * sum((weight_of_word(cw, lst_of_dicts) * dic[cw]) for cw in complex_words) / n_words
    else:
        print("Invalid mode!")

    fog_value = (words_per_sentence + percent_of_complex_words) * 0.4

    return {
        'fog': fog_value,  # wanted result
        'cws': complex_words,  # for check use
        'cwp': percent_of_complex_words,  # for check use
        'n_sents': n_sents,
        'n_words': n_words
    }


if __name__ == '__main__':
    # Important initialization
    filter_list = load_filter_list()
    lst_of_dicts = load_all(input("Please enter the directory of the documents: ")) # This will cd to the docs' directory
    with open('../fogs.csv', 'w+', encoding='UTF-8-sig', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['Title', 'Original fog', 'Weighted fog', 'Weighted fog (list)', 'n_sentences', 'n_words'])

        for file in os.listdir(os.curdir):
            original = fog(file, filter_list, lst_of_dicts, mode='o')
            weighted = fog(file, filter_list, lst_of_dicts, mode='w')
            weighted_plus_list = fog(file, filter_list, lst_of_dicts, mode='w+l')
            print('Title: ' + file)
            print('Original fog: ----   ' + str(original['fog']))
            print('Weighted fog: ----   ' + str(weighted['fog']))
            print('Weighted fog (list): ' + str(weighted_plus_list['fog']))
            # write data into table
            row = [file, original['fog'], weighted['fog'], weighted_plus_list['fog'], original['n_sents'], original['n_words']]
            writer.writerow(row)


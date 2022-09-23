import string
import numpy as np

def extract_word(input_string):

    # TODO: Implement this function
    for c in string.punctuation:
        input_string = input_string.replace(c, ' ')
    return list(filter(None, input_string.lower().split(' ')))


def extract_dictionary(df):
    word_dict = {}
    # TODO: Implement this function
    idx = 0
    for i in df.index:
        words = extract_word(df['text'][i])
        for w in words:
            if w in word_dict:
                continue
            word_dict[w] = idx
            idx += 1
    return word_dict


def generate_feature_matrix(df, word_dict):
    """Create matrix of feature vectors for dataset.

    Reads a dataframe and the dictionary of unique words to generate a matrix
    of {1, 0} feature vectors for each review.  Use the word_dict to find the
    correct index to set to 1 for each place in the feature vector. The
    resulting feature matrix should be of dimension (# of reviews, # of words
    in dictionary).

    Input:
        df: dataframe that has the text and labels
        word_dict: dictionary of words mapping to indices
    Returns:
        a numpy matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    # TODO: Implement this function
    count = 0
    for i in range(number_of_reviews):
        words = extract_word(df['text'][i])
        for w in words:
            feature_matrix[i][word_dict[w]] = 1
            count += 1
    print(count / number_of_reviews)
    return feature_matrix




from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
mystem = PorterStemmer()
lem = WordNetLemmatizer()


def construct_bag_of_words(input_sentence_tok,toknzd_all_words):

    # bag of words takes an input sentence and
    # tokenize sentence,
    # stem sentence
    # find matching peers in the tokenized pattern and  stem sentence
        input_text_strings = []
        #input_text_strings=[mystem.stem(words) for words in input_sentence_tok]#nltk.word_tokenize(input_sentence_tok)]
        mystop_words = stopwords.words('english')
        for words in input_sentence_tok:
            if words not in mystop_words:  # New Line included on the 21 Dec
        #for words in nltk.word_tokenize((input_sentence_tok)):
            #input_text_strings.append(mystem.stem(words))
                input_text_strings.append(lem.lemmatize(words))
        #print(input_text_strings)
        word_bag_container = np.zeros(len(toknzd_all_words),dtype=np.float32) # create a bag with the length of the tokenized bag words from patterns
        for index, words in enumerate(toknzd_all_words):
            if words in input_text_strings:
                word_bag_container[index] = 1.0
        return word_bag_container



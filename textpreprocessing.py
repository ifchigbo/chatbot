import json
from nltk.stem.porter import PorterStemmer
import nltk
from string import punctuation
from bag_of_words import construct_bag_of_words
import numpy as np
#updates done on the 9 of Dec
from nltk.stem import WordNetLemmatizer
#end of update 1
from nltk.corpus import stopwords

#file_path = './intents.json'
file_path = './newintents.json'

with open(file_path, mode='r') as knowledgebase:
    myfile = json.load(knowledgebase)

all_tags = [] # all tags - that deals with category of intents
word_patterns_tokenized = [] # all the list of tokenized words
word_patterns_tokenized_processed = []
all_response = [] # all responses from the chat bot
xyTrainingSet = [] # our X and Y training set that will be fed into the AI system
punct_special_char =[] # list of special characters and ignore statements
lem_words = [] #stemmertized words - words in their root form

#Deal with Tokenization and paring tags and tokens
for intent in myfile['intents']:
    all_tags.append(intent['tag'])
    for patten in intent['patterns']:
        word_patterns_tokenized.extend(nltk.word_tokenize(patten))
        xyTrainingSet.append((nltk.word_tokenize(patten),intent['tag']))


def getListofSpecialChars(): # get all special chars
    try:
        for chars in punctuation:
            punct_special_char.append(chars)
    except BaseException as error:
        print("The following Error has occured: {}".format(error))

def cleanPreprocessedData(): # clean tokenized data set
    getListofSpecialChars()
    try:
        for words in word_patterns_tokenized:
            if words not in punct_special_char:
                word_patterns_tokenized_processed.append(words)
    except BaseException as error:
        print("The following Error has occured")


# i will edit the whole section of this file for test - 6th December 2022
#def stemWordsLower():
#    cleanPreprocessedData()
#    stem = PorterStemmer()
#    for items in word_patterns_tokenized_processed:
#        stemmed_words.append(stem.stem(items).lower())

# Call the Stemmed Words and sort, takiing only unique values of the sorted words"
def lemWordsLower():
    cleanPreprocessedData()
    lem = WordNetLemmatizer()
    for items in word_patterns_tokenized_processed:
        lem_words.append(lem.lemmatize(items).lower())
lemWordsLower()

# -- End of edit -  6th december 2022

#sorted Tags
all_tags = sorted(set(all_tags))
#stemmed_words = sorted(set(lem_words))
_lem_words = sorted(set(lem_words))
#sorted Tags ends here
#Holding variables for the chatbot training data sets

# New Line start
stop_words = stopwords.words('english')

lem_words = []
for words in _lem_words:
    if words not in stop_words:
        lem_words.append((words))
# New line code block end


X_train = []
Y_train = []

for (pattern_sentence, tags) in xyTrainingSet: #Calling tokenized pattern above, and the tags for each pattern
    converted_bags_features = construct_bag_of_words(pattern_sentence, lem_words)
    X_train.append(converted_bags_features)
    label = all_tags.index(tags)
    Y_train.append(label)


X_train = np.array(X_train)
Y_train = np.array(Y_train)





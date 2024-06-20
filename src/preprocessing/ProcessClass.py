from src.constants import *
from nltk.tokenize import word_tokenize
import string
import contractions
from tokenize_all import Java
import re


class ProcessClass:
    def __init__(self, data):
        self.data = data
        self.processed_data = None

    # Function to preprocess text
    # @return: list of preprocessed and filtered tokens
    def process(self):
        pass


class CodePreprocessor(ProcessClass):
    # Need to know the programming language to preprocess code
    def __init__(self, data, language):
        super().__init__(data)
        self.language = language

    def python_preprocess(self):
        pass

    done = False

    def java_preprocess(self):
        # Remove comments
        remove_comments = re.sub(r"//.*", "", self.data)

        # Remove whitespace
        remove_whitespace = remove_comments.replace("\n", "").replace("\t", "")

        # Tokenize the code
        tokens = Java.tokenize(remove_whitespace)
        return tokens

    def process(self):
        if self.language == "python":
            self.python_preprocess()
        elif self.language == "java":
            self.processed_data = self.java_preprocess()
            return self.processed_data
        else:
            raise ValueError("Language not supported")


class TextPreprocessor(ProcessClass):
    def __init__(self, data, set_of_stopwords, lemmatiser, rake_object):
        super().__init__(data)
        self.set_of_stopwords = set_of_stopwords
        self.lemmatiser = lemmatiser
        self.rake_object = rake_object

    def process(self):
        # Tokenize the text
        expanded_text = contractions.fix(self.data)

        # Extract keywords from the text
        key_phrases = self.rake_object.process_text(expanded_text)

        # Remove punctuation and special characters using regex, because Rake doesn't remove all punctuations.
        # Source: https://www.geeksforgeeks.org/string-punctuation-in-python/
        key_phrases = [
            re.sub(rf"[{re.escape(string.punctuation)}]", "", key_phrase)
            for key_phrase in key_phrases
        ]

        # Get tokenize the words in the key phrases, each key phrase is now a list of keywords
        key_phrases = [word_tokenize(key_phrase) for key_phrase in key_phrases]

        # Flatten the list of lists
        keywords = [keyword for key_phrase in key_phrases for keyword in key_phrase]

        # Next, we will lemmatise the tokens to reduce the number of unique words
        # Source: https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
        keywords = [self.lemmatiser.lemmatize(keyword) for keyword in keywords]

        # Remove 's from the tokens
        keywords = [keyword for keyword in keywords if keyword != "'s"]

        # Lower case the tokens
        keywords = [keyword.lower() for keyword in keywords]

        return keywords

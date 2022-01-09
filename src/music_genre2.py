from nltk import text
from bs4 import BeautifulSoup
import unicodedata
import contractions
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
import re
import string
import emoji

from nltk.corpus import stopwords, wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect, DetectorFactory
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('wordnet') 
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

class Norm_corpus:
    '''
    Generalized Corpus Norming Class
    Depends on:
        bs4's Beautiful Soup (HTML parser)
        NLTK
            averaged perceptron tagger for Lemmitization Parts of Speech tagging
            stopwords
            wordnet
            stem (WordNetLemmatizer)
        re for regular expressions
        emoji for replacing emojis and emoticons in text (if requested)

    '''
    def __init__(self, demoji = False, strip_html=True, strip_entities = True, delink = True, unaccent=True, 
                 lower = True, decontract=True, lemma = False, unspecial = True, undigit = False, unstop = True,
                lang = 'english', no_strip = ["'"]):
        self.text = text
        self.demoji = demoji
        self.delink = delink
        self.strip_html = strip_html
        self.strip_entities = strip_entities
        self.unaccent=unaccent
        self.lower = lower
        self.decontract = decontract
        self.lemma = lemma
        self.unspecial = unspecial
        self.undigit = undigit
        self.unstop = unstop
        self.lang = lang # language to use, or 'detect'
        self.no_strip = no_strip
    
    def fit(self, corpus):
        '''
        Inspired by: 
        https://towardsdatascience.com/
        a-practitioners-guide-to-natural-language-processing
        -part-i-processing-understanding-text-9f4abfd13e72
        '''
        
        normed_corpus = []
        for doc in corpus:
            # strip html
            if self.strip_html:
                doc = self.strip_html_tags(doc)
            if self.demoji:
                doc = self.replace_emoji(doc)
            if self.delink:
                doc = self.strip_links(doc)
            if self.lang == "detect":
                self.language = self.lang_select(doc)
            else: 
                self.language = self.lang
            if self.unaccent:
                doc = self.remove_accents(doc)
            if self.lower:
                doc = doc.lower()
            if self.decontract:
                doc = self.expand_contractions(doc)
            if self.strip_entities:
                doc = self.strip_all_entities(doc)
            # remove extra newlines
            doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
            if self.lemma:
                doc = self.lemmatize(doc)
            if self.unspecial:
                # insert spaces between special characters to isolate them    
                special_char_pattern = re.compile(r'([{.(-)!}])')
                doc = special_char_pattern.sub(" \\1 ", doc)
                doc = self.remove_special(doc)
            if self.undigit:
                doc = self.remove_digits(doc)
            # remove extra whitespace
            doc = re.sub(' +', ' ', doc)
            if self.unstop:
                doc = self.remove_stopwords(doc)
        normed_corpus.append(doc)
        return normed_corpus

    def expand_contractions(self, text):
        expanded_words = []    
        for word in text.split():
            # using contractions.fix to expand the shotened words
            expanded_words.append(contractions.fix(word))   
                
        return ' '.join(expanded_words)

    def lang_select(self, doc):
        lang = ''
        DetectorFactory.seed = 0
        lang = detect(doc)
        print(lang)
        return lang
    
    def replace_emoji(self, text):
        return emoji.demojize(text, delimiters=("", ""))
    
    def strip_html_tags(self, doc):
        '''
        Via: https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i
             -processing-understanding-text-9f4abfd13e72
        Uses BeautifulSoup library's HTML parser
        '''
        soup = BeautifulSoup(doc, "html.parser")
        stripped_text = soup.get_text()
        return stripped_text       

    def strip_all_entities(self, text):
        '''
        Via https://stackoverflow.com/questions/8376691/
            how-to-remove-hashtag-user-link-of-a-tweet-using-regular-expression
        '''
        entity_prefixes = self.no_strip
        for separator in  string.punctuation:
            if separator not in entity_prefixes:
                text = text.replace(separator,' ')
        words = []
        for word in text.split():
            word = word.strip()
            if word:
                if word[0] not in entity_prefixes:
                    words.append(word)
        return ' '.join(words)

    def remove_accents(self, doc):
        text = unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def decontract(self, text):
        expanded_words = list()
        for word in text.split():
            expanded_words.append(contractions.fix(word))
        return ' '.join(expanded_words)

    def lemmatize(self, doc):
        lemmatizer = WordNetLemmatizer()
        text = [lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) for word in nltk.word_tokenize(doc)]
        return ' '.join(text)

    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def strip_links(self, text):
        '''
        Via https://stackoverflow.com/questions/8376691/
            how-to-remove-hashtag-user-link-of-a-tweet-using-regular-expression
        '''
        link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', 
            re.DOTALL)
        links = re.findall(link_regex, text)
        for link in links:
            text = text.replace(link[0], ', ')    
        return text

    def remove_special(self, text):
        pattern = r'[^a-zA-z0-9\s]'
        text = re.sub(pattern, '', text)
        return text

    def remove_stopwords(self, text):
        '''
        Inspired by:
        https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
        '''
        try:
            stop_words = set(stopwords.words(self.language))
            word_tokens = word_tokenize(text)
            filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
            filtered_sentence = []
            for w in word_tokens:
                if w not in stop_words:
                    filtered_sentence.append(w)
            return filtered_sentence
        except:
            if self.language == "tl":
                pass

    def save(obj):
        # For package pickle introspection
        return (obj.__class__, obj.__dict__)

    def load(cls, attributes):
        # For package pickle introspection  
        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj
from nltk import text
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
import re
import string
import re
import emoji

from nltk.corpus import stopwords, wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect, DetectorFactory
from nltk.stem import SnowballStemmer, WordNetLemmatizer

nltk.download('wordnet') 
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')



class TweetGenreList:
    '''
    Performs data cleaning and, optionally, NLTK stemming and lemmitization prior to 
    vader sentiment analysis on files of tweets containing genre text.
    '''
    def __init__(self, filepath, filepathmeta, genres, exclude_re, 
                 type = [None, 'stem', 'lemma'], lang = ['google', 'langdetect']):
        # Initialize
        self.filepath = filepath
        self.filepathmeta = filepathmeta
        self.genres = genres
        self.exclude_re = exclude_re
        self.type = type
        self.lang_ = lang
        #   Get df & clean
        self.df = self.get_datafile()
        self.n_total = self.df.shape[0]
        self.df1, self.df2, self.df3 = self.reduce()
        self.issolate_records()

    def percentage(self, part, whole):
        return 100*float(part)/float(whole)

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


    def strip_all_entities(self, text):
        '''
        Via https://stackoverflow.com/questions/8376691/
            how-to-remove-hashtag-user-link-of-a-tweet-using-regular-expression
        '''
        entity_prefixes = ['@','#',"'"]
        for separator in  string.punctuation:
            if separator not in entity_prefixes :
                text = text.replace(separator,' ')
        words = []
        for word in text.split():
            word = word.strip()
            if word:
                if word[0] not in entity_prefixes:
                    words.append(word)
        return ' '.join(words)


    def replace_emoji(self, text):
        return emoji.demojize(text, delimiters=("", ""))
    
    
    def clean(self, df):
        index = df["text"].str.contains(self.exclude_re, 
            flags=re.IGNORECASE, regex=True)
        df = df[~index].copy()
        df = df.reset_index()
        #   Strip Emojis & Emoticons
        df = self.de_emoji(df)
        return df

        
    def get_datafile(self):
        df = pd.read_csv(self.filepath, compression="gzip", low_memory=True)
        df.drop_duplicates(inplace=True, subset="id")
        df = self.clean(df)

        return df


    def de_emoji(self, df):
        new_row = list()
        for row in df["text"]:
            #get rid of hashtags, mentions, links, emojis, and emoticons
            new_row_ = self.strip_links(row)
            new_row_ = self.strip_all_entities(new_row_)
            new_row_ = self.replace_emoji(new_row_)
            new_row.append(new_row_)
        df['stripped'] = new_row
        df.drop_duplicates(inplace = True, subset="stripped")
        return df.copy()


    def write_reduced(self, df2, df1, compression = None):
        self.filepathreduced  = self.filepath[0:-4]+".results.csv"
        self.filepathreduced2 = self.filepath[0:-4]+".results2.csv"
        print("Outfile 1: " + self.filepathreduced)
        print("Outfile 2: " + self.filepathreduced2)

        df2.to_csv(self.filepath[0:-4]+".reduced.csv", compression = compression, index = False)
        df1.to_csv(self.filepath[0:-4]+".reduced2.csv", compression = compression, index = False)
        return None



    def reduce(self):
        df3 = pd.DataFrame({self.n_total:np.array(["len_genre","len_pos","len_neg","len_neut"])})
        df2 = pd.DataFrame({"sum_":np.zeros(shape=self.df.shape[0])})
        df1 = pd.DataFrame({"sum_":np.zeros(shape=self.df.shape[0])})
        for genre in genre_list:
            print("GENRE: ", genre, "\n")
            index = self.df["stripped"].str.contains(genre, flags=re.IGNORECASE, regex=True)
            df2[genre] = index 
            df1[genre] = self.df.loc[index,'stripped'].copy()
        for row in range(df2.shape[0]):
            df2.iloc[row, 0] = np.sum(np.array(df2.iloc[row, 1:], dtype=float))
            df1["sum_"]=df2["sum_"]
        self.write_reduced(df2, df1)
        return df1, df2, df3


    def issolate_records(self):
        idx = self.df1.loc[:,"sum_"]==1
        summy = np.sum(np.array(idx, dtype=float))
        print(str(round(summy,2)) + " issolated records")
        self.df1 = self.df1.loc[idx, :].copy()
        print(self.df1.info())
        df4 = pd.DataFrame()
        for column in genre_list:
            d = self.df1.loc[self.df1.loc[:, column].notna(),column].copy()
            d.drop_duplicates(inplace=True)
            print(d)
            lengthy = d.shape[0]
            if lengthy > 0:
                self.df3[column], dd = self.main(d, column, lengthy)
                df4.append(dd)
        self.df4 = df4
        self.write_results()


    def write_results(self, compression = None):
        if self.type == 'stem':
            self.filepathresults = self.filepath[0:-4]+".results.stem.csv"
            self.filepathresults2= self.filepath[0:-4]+".tweets.stem.csv"
        elif self.type == 'lemma':
            self.filepathresults = self.filepath[0:-4]+".results.lemma.csv"
            self.filepathresults2= self.filepath[0:-4]+".tweets.lemma.csv"
        else:
            self.filepathresults = self.filepath[0:-4]+".results.csv"
            self.filepathresults2= self.filepath[0:-4]+".tweets.csv"
        self.df3.to_csv(self.filepathresults, compression=compression, index=False)
        self.df4.to_csv(self.filepathresults2, compression=compression, index=False)
        return None



    def lang_select(self, tweet):
        lang = ''
        if self.lang_ == 'google':
            tb = TextBlob(tweet)
            lang = tb.detect_language()
        else:
            DetectorFactory.seed = 0
            lang = detect(tweet)
        return lang
        
    def remove_stops(self, tweet):
        #lang = self.lang_select(tweet)
        stop = set(stopwords.words("english"))
        text = [word for word in tweet.lower().split() if word not in stop]
        return ' '.join(text)


    def stem_tweet(self, tweet):
        '''
        stem_tweet() uses the SnowballStemmer(). Autodetect language not working.
        Appears to break pos tags as well?
        '''
        #lang = self.lang_select(tweet)
        stemmer = SnowballStemmer(language = "english", ignore_stopwords=False)
        return ' '.join(stemmer.stem(tweet))

    def lemma_tweet(self, tweet):
        lemmatizer = WordNetLemmatizer()
        text = [lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) for word in nltk.word_tokenize(tweet)]
        return ' '.join(text)

    def main(self, tweets, genre, n_total):
        scorer = []
        positive = 0
        negative = 0
        neutral = 0
        polarity = 0
        tweet_list = []
        neutral_list = []
        negative_list = []
        positive_list = []
        compound_list = []
        for tweet in tweets:
            text = ""
            tweet = self.remove_stops(tweet)
            
            if self.type == 'stem':
                text = self.stem_tweet(tweet)
            elif self.type == 'lemma':
                text = self.lemma_tweet(tweet)
            else:
                text = tweet
            tweet_list.append(text)
            analysis = TextBlob(text)
            score = SentimentIntensityAnalyzer().polarity_scores(text)
            scorer.append(score)
            neg = score['neg']
            neu = score['neu']
            pos = score['pos']
            comp = score['compound']
            polarity += analysis.sentiment.polarity
            compound_list.append(comp)
            if neg > pos:
                negative_list.append(text)
                negative += 1
            elif pos > neg:
                positive_list.append(text)
                positive += 1
                
            elif pos == neg:
                neutral_list.append(text)
                neutral += 1
        positive = self.percentage(positive, n_total)
        negative = self.percentage(negative, n_total)
        neutral = self.percentage(neutral, n_total)
        polarity = self.percentage(polarity, n_total)
        positive = format(positive, '.1f')
        negative = format(negative, '.1f')
        neutral = format(neutral, '.1f')
    
        tweet_list = pd.DataFrame({"tweet_list":tweet_list, "score":scorer})
        neutral_list = pd.DataFrame(neutral_list)
        negative_list = pd.DataFrame(negative_list)
        positive_list = pd.DataFrame(positive_list)
        print("total number: ",len(tweet_list))
        print("positive number: ",len(positive_list))
        print("negative number: ", len(negative_list))
        print("neutral number: ",len(neutral_list))
        print("\n\n\n")
        return pd.Series(np.array([tweet_list.shape[0],len(positive_list),len(negative_list),
            len(neutral_list)])), tweet_list

if __name__ == "__main__":
    genre_list = np.array(['jazz(\W|\s)','metal(\W|\s)','opera(\W|\s)',
        'folk(\W|\s)','indie(\W|\s)','[^indie-]rock(\W|\s)',
        'trance(\W|\s)','blues(\W|\s)','grunge(\W|\s)','classical|chamber|orchestra(\W|\s)',
        'funk(\W|\s)','emo(\W|\s)','reggae|regae(\W|\s)','country|western|westren|cw|c-w|c amp w|c w(\W|\s)',
        'k-pop|kpop|k pop(\W|\s)','[^k-]pop(\W|\s)','techno(\W|\s)',
        'edm|electro(\W|\s)','r amp b|r&amp;b|rnb|randb|r b|r and b(\W|\s)','house(\W|\s)',
        'hip hop|hiphop|hip-hop(\W|\s)','punk(\W|\s)',
        'dubstep|dub step|dubs|dub|dubz(\W|\s)','rap(\W|\s)',
        'drum and bass|drumnbass|dnb|drum amp bass|d amp b|d b|d n b|drumandbass|dandb|drum n bass(\W|\s)',
        'grime(\W|\s)','drill(\W|\s)'])
    filepath = "tweet27.csv"
    filepathmeta = None
    exclude_re = "RT|out now|outnow|exclusive|new release|newrelease|now playing|nowplaying|official|COMMERCIAL &amp; TALK FREE"
    
    # Instantiate Object
    # Stemming appears to be broken
    data_obj = TweetGenreList(filepath, filepathmeta, genre_list, exclude_re, type="stem", lang = 'langdetect')
    # lemmatization works fine with pos tags
    data_obj2 = TweetGenreList(filepath, filepathmeta, genre_list, exclude_re, type="lemma", lang = 'langdetect')
    # Slight differences between lemmitization and no text adjustment before sentiment analysis
    data_obj3 = TweetGenreList(filepath, filepathmeta, genre_list, exclude_re, type = None, lang = 'langdetect')




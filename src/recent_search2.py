import requests
import os
import pandas as pd
import time
import numpy as np
from numpy.random import Generator, PCG64
import pickle


class GetGenreTweet2_0:
    '''
    Class GetGenreTweet2_0 is a wrapper for a few of the Twitter API 2.0 Recent Search functions
    with added functionality for acquiring genre data. The call to the API results in many duplicates, 
    and therefore a larger file that gets scrubbed in data processing with 
    index = df['id'].drop_duplicates().

    Data are stored in gzip compressed csv files. 
    '''
    def __init__(self, 
                 filepath = "tweetv2_0.csv", 
                 filepathmeta = "tweetv2_0meta.csv",
                 search_url = "https://api.twitter.com/2/tweets/search/recent",
                 # To set your environment variables in your terminal run the following line:
                 # export 'BEARER_TOKEN'=''
                 bearer_token = None,
                 genres = np.array(['jazz','metal','opera','folk','indie','rock','trance','blues',
                    'grunge', 'classical','funk','emo','reggae','country','k-pop','pop','techno','kpop',
                    'edm','electronic','electronica','randb','rnb','r&amp;b','house','hip hop','hiphop',
                    'hip-hop', 'punk','dubstep','rap','drum and bass', 'dnb','drumnbass','drum&amp;bass',
                    'grime','drill', 'k pop', 'r and b', 'd&amp;b']),
                 # Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
                 # expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
                 maxresults = 80,
                 query_params = None):
        self.filepath = filepath
        self.filepathmeta = filepathmeta
        self.search_url = search_url
        try:
            if bearer_token is None:
                self.bearer_token = os.environ.get("BEARER_TOKEN")
            else:
                self.bearer_token = bearer_token
        except:
            print("No OS BEARER_TOKEN found.")
            print("Must set first by calling: export 'BEARER_TOKEN'='enter your Twitter API v2.0 token here'")
        self.genres = genres
        self.rng = Generator(PCG64(seed=None))
        self.randomized_genres = self.rng.choice(genres, size=genres.shape[0], replace=False)                   
        self.genre = self.randomized_genres[0]
        self.maxresults = maxresults
        if query_params is None:
            self.query_params = self.return_query()
        else:
            self.query_params = query_params
                
    def return_query(self, dictionary = None):
        if dictionary is None:
            dictionary = {'query':f'-is:retweet music {self.genre} lang:en -RT -"RT" -politics -government -#tunein -live -@mixcloud -"now available" -Premiere -"Top 100" -radio -#NowPlaying -"now playing" -@YouTube -Favorited -@phoenix_rec -#Spotify -subscribe -"available now" -vote -"chance to win" -WIN',
                                 'tweet.fields': 'author_id,text,context_annotations', 
                                 'user.fields':'created_at,profile_image_url,username',
                                 'place.fields':'country',
                                 'max_results':f'{self.maxresults}'}
        return dictionary

    def bearer_oauth(self, r):
        # From Twitter API 2.0 
        """
        Method required by bearer token authentication.
        """

        r.headers["Authorization"] = f"Bearer {self.bearer_token}"
        r.headers["User-Agent"] = "v2RecentSearchPython"
        return r

    def connect_to_endpoint(self):
        # From Twitter API 2.0 
        response = requests.get(self.search_url, auth=self.bearer_oauth, params=self.query_params)
        print(response.status_code)
        if response.status_code != 200:
            raise Exception(response.status_code, response.text)
        return response.json()
    
    def get_first(self):
        '''
        If the files don't yet exist, initialize the dataframes
        with a first call to the Twitter API 2.0.
        '''
        
        json_response = self.connect_to_endpoint()
        df = pd.json_normalize(json_response['data'])
        df2 = pd.json_normalize(json_response['meta'])
        return df, df2

    def read_previous(self):
        '''
        Read the previously stored data.
        '''
        try:
            df = pd.read_csv(self.filepath, compression="gzip", sep=",", low_memory=True)
            df2= pd.read_csv(self.filepathmeta, compression="gzip", sep=",", low_memory=True)
        except:
            df, df2 = self.get_first()
        self.df = df
        self.df2 = df2
        return None

    def get(self, max_records_for_df, save_interrupt = 1, n_requests = 50, maxresults = 80, minutes_between = 4):
        # Initialize df
        self.maxresults = maxresults
        self.read_previous()
        saver = 0
        while (self.df.shape[0] <= max_records_for_df) & (self.df.shape[0] > 0):
            self.get_tweets(n_requests, minutes_between)
            if self.df.shape[0] % save_interrupt*int(n_requests/self.maxresults) == 0:
                self.df.to_csv(self.filepath+"."+str(saver), sep=",", compression="gzip", index=False)
                self.df2.to_csv(self.filepathmeta+"."+str(saver), sep=",", compression="gzip", index=False)
                saver+=1      
        self.df.to_csv(self.filepath, sep=",", compression="gzip", index=False)
        self.df2.to_csv(self.filepathmeta, sep=",", compression="gzip", index=False)


    def get_tweets(self, n_requests, minutes_between):
        '''
        Get_tweets is the low level workhorse function. It re-randomizes every time
        get_tweets is called, then loops through the range of requests, trying each set.
        '''
        self.rng = Generator(PCG64(seed=None))
        self.randomized_genres = self.rng.choice(self.genres, size=self.genres.shape[0], replace=False)
        len_genres = self.randomized_genres.shape[0]
        for record in range(int(n_requests/len_genres)):
            for self.genre in self.randomized_genres:
                self.query_params = self.return_query()

                json_response = self.connect_to_endpoint()
                try:
                    self.df = self.df.append(pd.json_normalize(json_response['data']))
                    self.df2 = self.df2.append(pd.json_normalize(json_response['meta']))
                    print(f"New Data: {self.genre}")
                    print(self.df.tail(5)['id'])
                    print("\n\n")
                    print(self.query_params)
                    print("\n\n\n\n")
                except:
                    print("\tError - Skipped")
                    pass
                time.sleep(minutes_between*60)
        return None

    def print_data(self, type=["query","tail","head", "genres"], data=['id','text']):
        output = {"query":self.query_params, 
                  "tail":self.df.tail(5)[data], 
                  "head":self.df.head(5)[data], 
                  "genres":self.genres}

        print(output[type])
        return None
     
    def save(obj):
        # For package pickle introspection
        return (obj.__class__, obj.__dict__)

    def load(cls, attributes):
        # For package pickle introspection  
        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj


    def set_data(self, type=np.array(['filepath', 'filepathmeta', 'query', 'genres']), data=None):
        output = {'filepath': 'self.filepath = data',
                  'filepathmeta': 'self.filepathmeta = data',
                  'query': 'self.query_params = data',
                  'genres': 'self.genres = data'}
        exec(output[type])
        return None


if __name__ == "__main__":
    # Test get on small sample then pickle
    t2_0 = GetGenreTweet2_0(filepath="new.csv", filepathmeta="newmeta.csv")
    t2_0.get(1, 1, 1, 80, 4)
    file_obj_w = open("t2_0.obj", "wb")
    pickle.dump(t2_0, file_obj_w, protocol=4)
    
    # Test unpickle then set functions
    file_obj_r = open("t2_0.obj", "rb")
    t2_02 = pickle.load(file_obj_r)
    t2_02.print_data("tail")
    t2_02.set_data(type="genres", data="trance")
    t2_02.print_data("genres")

    t2_0.get(60_000, 1, 80 * t2_0.genres.shape[0], 80, 4)



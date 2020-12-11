import tweepy
from tweepy import OAuthHandler
from tweepy import Cursor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, date, time, timedelta
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
#import credentials
import streamlit as st
from decouple import config
import os


@st.cache(allow_output_mutation=True,suppress_st_warning=True,show_spinner=False)
def get_user(searchQuery):
    user = api.user_timeline(screen_name=searchQuery, count=200)
    u = []
    for obj in user:
        u.append(obj)
    return u

@st.cache(allow_output_mutation=True,suppress_st_warning=True,show_spinner=False)
def get_user_hashtags(target):
    print("Getting hashtags for " + target)
    item = api.get_user(target)
    hashtags = []
    count_hashtags = []
    tweet_count = 0
    end_date = datetime.utcnow() - timedelta(days=14)
    for status in Cursor(api.user_timeline, id=target).items():
        tweet_count += 1
        if hasattr(status, "entities"):
            entities = status.entities
        if "hashtags" in entities:
            for ent in entities["hashtags"]:
                if ent is not None:
                    if "text" in ent:
                        hashtag = ent["text"]
                    if hashtag is not None:
                        hashtags.append(hashtag)

        if status.created_at < end_date:
            break
    hashtags_final = []
    for item, count in Counter(hashtags).most_common(500):
        hashtags_final.append(item + " " + str(count))
        count_hashtags.append(count)

    return hashtags_final


@st.cache(allow_output_mutation=True,suppress_st_warning=True,show_spinner=False)
def get_user_mentions(target):
    print("Getting mentions for " + target)
    item = api.get_user(target)
    mentions = []
    tweet_count = 0
    end_date = datetime.utcnow() - timedelta(days=30)
    for status in Cursor(api.user_timeline, id=target).items():
        tweet_count += 1
        if hasattr(status, "entities"):
            entities = status.entities

        if "user_mentions" in entities:
            for ent in entities["user_mentions"]:
                if ent is not None:
                    if "screen_name" in ent:
                        name = ent["screen_name"]
                    if name is not None:
                        mentions.append(name)
        if status.created_at < end_date:
            break
    mentions_final = []
    for item, count in Counter(mentions).most_common(500):
        mentions_final.append(item + " " + str(count))
    return mentions_final


def get_hashtags_mentions(search):
    hashtags_list = get_user_hashtags(search)
    mentions_list = get_user_mentions(search)
    hashtags = {}
    for word in hashtags_list:
        words = word.split()
        hashtags[words[0]] = int(words[1])
    mentions = {}
    for word in mentions_list:
        words = word.split()
        mentions[words[0]] = int(words[1])
    return hashtags, mentions


def is_match(st):

   import re
   if (re.findall(r'bot\b', st, flags=re.IGNORECASE)) or (re.findall(r'b0t\b', st, flags=re.IGNORECASE)) :
       return True
   else:
    return False

def bot_in_name(user):
    if ((is_match(str(user['name']))==True) or (is_match(str(user['screen_name']))==True) or (is_match(str(user['description']))==True) ):

        return True
    else:
        return False




def get_score(user):
    score = user['log_friends_to_followers']
    score = score + user['log_friends_growth_rate'] - user['log_followers_growth_rate']
    score = score - user['log_listed_count']

    score = score + user['status_to_age_ratio']/72

    if user['description_length'] <4:

        score += 3
    if user['verified']==True:
        score -= 20

    if user['default_profile']==True:
       score += 3

    if user['geo_enabled']==True:
        score += 3

    score = score + user['screen_name_digits']/3
    return score

def process_results(results):
    id_list = [tweet.id for tweet in results]
    data_set = pd.DataFrame(id_list, columns=["id"])

    # Processing Tweet Data

    data_set["text"] = [tweet.text for tweet in results]  # text of tweet
    data_set["probe_timestamp"] = [tweet.created_at for tweet in results]  # when the tweet was created
    data_set['hour'] = data_set['probe_timestamp'].dt.hour
    data_set['weekday'] = data_set["probe_timestamp"].dt.day_name()
    data_set["retweet_count"] = [tweet.retweet_count for tweet in results]  # number of retweets
    data_set["favorite_count"] = [tweet.favorite_count for tweet in results]  # number of favourites
    data_set["source"] = [tweet.source for tweet in results]  # source of the tweet
    data_set["length"] = [len(tweet.text) for tweet in results]  # number of characters in tweet

    # Processing User Data
    data_set["id"] = [tweet.author.id for tweet in results]  # id of the author
    data_set["screen_name"] = [tweet.author.screen_name for tweet in results]
    data_set["name"] = [tweet.author.name for tweet in results]
    data_set["user_created_at"] = [tweet.author.created_at for tweet in results]  # age of user account
    data_set["description"] = [tweet.author.description for tweet in results]
    data_set['description_length'] = [len(tweet.author.description) for tweet in results]
    data_set["followers_count"] = [tweet.author.followers_count for tweet in results]  # number of followers
    data_set["friends_count"] = [tweet.author.friends_count for tweet in results]  # number of friends
    data_set["location"] = [tweet.author.location for tweet in results]  # user has a location in profile?
    data_set["statuses_count"] = [tweet.author.statuses_count for tweet in results]  # number of statuses
    data_set['favourites_count'] = [tweet.author.favourites_count for tweet in results]
    data_set['listed_count'] = [tweet.author.listed_count for tweet in results]
    data_set['default_profile'] = [tweet.author.default_profile for tweet in results]
    data_set["verified"] = [tweet.author.verified for tweet in results]  # user is verified?
    data_set["url"] = [tweet.author.url for tweet in results]  # user has a URL?
    data_set['geo_enabled'] = [tweet.author.geo_enabled for tweet in results]
    data_set['friends_to_followers'] = data_set['friends_count'] / (data_set['followers_count'] + 1)
    data_set['popularity'] = data_set['followers_count'] / (data_set['followers_count'] + data_set['friends_count'] + 1)
    data_set["probe_timestamp"] = data_set['probe_timestamp'].map(lambda ts: ts.strftime("%Y-%m-%d"))

    data_set['user_created_at'] = pd.to_datetime(data_set['user_created_at'])
    data_set["date_user_created_at"] = data_set['user_created_at'].map(lambda ts: ts.strftime("%Y-%m-%d"))

    d = (pd.to_datetime(data_set['probe_timestamp']) - pd.to_datetime(data_set['date_user_created_at']))
    data_set['user_account_age'] = d.astype('timedelta64[D]')

    data_set['status_to_age_ratio'] = data_set['statuses_count'] / (data_set['user_account_age'] + 1)

    data_set['friends_growth_rate'] = data_set['friends_count'] / (data_set['user_account_age'] + 1)
    data_set['followers_growth_rate'] = data_set['followers_count'] / (data_set['user_account_age'] + 1)
    data_set['favourites_growth_rate'] = data_set['favourites_count'] / (data_set['user_account_age'] + 1)
    data_set['screen_name_digits'] = data_set['screen_name'].apply(lambda s: sum(c.isdigit() for c in s))

    data_set['log_followers_count'] = np.log((1 + data_set['followers_count']))
    data_set['log_friends_count'] = np.log((1 + data_set['friends_count']))
    data_set['log_favourites_count'] = np.log((1 + data_set['favourites_count']))
    data_set['log_listed_count'] = np.log((1 + data_set['listed_count']))
    data_set['log_statuses_count'] = np.log((1 + data_set['statuses_count']))
    data_set['log_description_length'] = np.log((1 + data_set['description_length']))
    data_set['log_friends_growth_rate'] = np.log((1 + data_set['friends_growth_rate']))
    data_set['log_followers_growth_rate'] = np.log((1 + data_set['followers_growth_rate']))
    data_set['log_favourites_growth_rate'] = np.log((1 + data_set['favourites_growth_rate']))
    data_set['log_status_to_age_ratio'] = np.log((1 + data_set['status_to_age_ratio']))
    data_set['log_friends_to_followers'] = np.log((1 + data_set['friends_to_followers']))
    data_set['bot_in_name'] = data_set.apply(bot_in_name, axis=1)
    data_set['bot_score'] = data_set.apply(get_score, axis=1)
    return data_set


def create_word_cloud(string):
    if len(string)==0:
        string = {' ':1}


    maskArray = np.array(Image.open("tweepy word cloud.png"))
    cloud = WordCloud(font_path='chandas1-2.ttf', background_color="white", max_words=200, width=1600, height=800,
                      mask=maskArray)
    cloud.generate_from_frequencies(string)

    return cloud


def get_plots(searchQuery, data, result, hashtags, mention):
    global ratio
    fig, ax = plt.subplots(figsize=(20, 10), ncols=3, nrows=2)
    fig.tight_layout(pad=6.0)
    ax[0, 0].set_title('Last {} tweets of {}'.format(len(data), searchQuery))
    graph = sns.countplot(x='weekday',
                          order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                          data=data, palette='viridis', ax=ax[0, 0])
    for p in graph.patches:
        height = p.get_height()
        graph.text(p.get_x() + p.get_width() / 2., height + 0.1, height, ha="center")
    ratio = data['text'].str.count('RT @').sum() / len(data)
    ax[0, 1].set_title('Last {} tweets of {}'.format(len(data), searchQuery))
    graph = sns.countplot(x='hour', data=data, ax=ax[0, 1])
    for p in graph.patches:
        height = p.get_height()
        graph.text(p.get_x() + p.get_width() / 2., height + 0.1, height, ha="center")

    retweet_ratio = data['text'].str.count('RT @').sum() / len(data)
    mentions = 0
    for text in data['text']:
        if '@' in text and 'RT' not in text:
            mentions += 1
    mentions_ratio = mentions / len(data)

    if retweet_ratio > 0.99:
        my_data = [retweet_ratio, (1 - retweet_ratio - mentions_ratio)]
        my_labels = ['retweet', 'mentions + Quotes']
        my_colors = ['#f6cd61', '#fe4a49']
        my_explode = (0.1, 0)
    elif (1 - retweet_ratio - mentions_ratio) > 0.99:
        my_data = [retweet_ratio + mentions_ratio, (1 - retweet_ratio - mentions_ratio)]
        my_labels = ['retweet + mentions', 'Quotes']
        my_colors = ['#f6cd61', '#fe4a49']
        my_explode = (0.1, 0)
    else:
        my_data = [retweet_ratio, mentions_ratio, (1 - retweet_ratio - mentions_ratio)]
        my_labels = ['retweet', 'mentions', 'Quotes']
        my_colors = ['#f6cd61', '#3da4ab', '#fe4a49']
        my_explode = (0.1, 0, 0)

    ax[1, 0].pie(my_data, labels=my_labels, autopct='%1.1f%%', startangle=15, shadow=True, colors=my_colors,
                 explode=my_explode)
    ax[1, 0].set_title('Tweet Distribution of {}'.format(searchQuery))
    ax[1, 0].axis('equal')

    ax[1, 1].text(0.4, 0.5, 'Result:{}\nBot Probability:{:.2f}%'.format(result[0], result[1]),
                  bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'), fontsize=20)
    ax[1, 1].axis('off')
    hash_cloud = create_word_cloud(hashtags)
    ax[0, 2].imshow(hash_cloud)
    ax[0, 2].set_title('Hashtag Cloud')
    ax[0, 2].axis('off')
    mentions_cloud = create_word_cloud(mention)
    ax[1, 2].imshow(mentions_cloud)
    ax[1, 2].set_title('Mentions Cloud')
    ax[1, 2].axis('off')
    plt.figure(figsize=(18, 20),dpi=300)
    #plt.show()
    st.pyplot(fig,figsize=(18, 20),dpi=300)

@st.cache(allow_output_mutation=True,suppress_st_warning=True,show_spinner=False)
def load_model():
    import pickle
    model_path = open('bot_scoreV1.3.pkl', 'rb')
    model = pickle.load(model_path)
    scaler = pickle.load(open('scalerV1.3.pkl', 'rb'))
    return  model,scaler

def get_prediction(searchQuery):
    user = get_user(searchQuery)
    data = process_results(user)

    use = ['default_profile', 'description_length','popularity',
           'log_listed_count', 'log_statuses_count', 'log_followers_count', 'log_friends_count',
           'log_favourites_count', 'log_followers_growth_rate', 'bot_score']

    df = data[use]
    model,scaler = load_model()
    X = df.iloc[:, :].values
    x = scaler.transform(X[1, :].reshape(1, -1))
    bot = ['Human', 'Bot']
    is_bot = bot[model.predict(x)[0]]
    percent_proba = model.predict_proba(x)[0, 1] * 100
    result = (is_bot, percent_proba)
    #result  = ('Yes',97)
    hastags, mentions = get_hashtags_mentions(searchQuery)
    get_plots(searchQuery, data, result, hastags, mentions)
    return data, result


if __name__ == '__main__':
    consumer_key = str(os.getenv('CONSUMER_KEY'))
    consumer_secret = str(os.getenv('CONSUMER_SECRET'))
    access_key = str(os.getenv('ACCESS_KEY'))
    access_secret =   str(os.getenv('ACCESS_SECRET'))

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)

    api = tweepy.API(auth)

    import time
    start_time = time.time()

    st.write("""
        ## Twitter Bot Detector
        
        """)
    search = st.text_input('Enter User')

    if st.button('Predict'):
        with st.spinner('Predicting.....'):
            st.subheader('Profile Analytics')
            data, result = get_prediction(search)
            st.subheader('Last {} tweets'.format(len(data)))
            st.write(data[['text','source',"followers_count","friends_count",'user_account_age']])
            st.subheader('Result')
            st.write(search, 'is a', result[0], 'and probability of being a bot is', result[1], '%')

    end_time = time.time()
    print("Total execution time: {} seconds".format(end_time - start_time))


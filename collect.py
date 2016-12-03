"""
collect.py
"""
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
import pickle
from twitter import *
from collections import Counter, defaultdict, deque
from itertools import chain, combinations
import os.path

consumer_key = 'PWy5WzvjT36uGAVHcM9AY5QHZ'
consumer_secret = 'VELshHQhruwxYQwpRHjE6OjG5yXjoCo1lYMWtOmZOwUpudmkNc'
access_token = '129882215-FDxcAMKqqzCctJwRx1szo2njSVTn5y7aaApuJzfC'
access_token_secret = 'XTIRz00BBPjJ6y73SRQcq8HkCLkSHbx4qMp1tkk6zcR1U'

def get_twitter(R):
    if R == "API":
        return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)
    else:
        return Twitter(auth = OAuth(access_token, access_token_secret, consumer_key, consumer_secret))


def robust_request(twitter, resource, params, max_tries=5):
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def get_train_tweets(twitter,username):
    tweets = []
    for r in robust_request(twitter,'statuses/filter', {'track': username},5):
        tweets.append(r)
        if len(tweets) % 100 == 0:
            print('%d tweets' % len(tweets))
        if len(tweets) >= 500:
            break
    print('Retrieved %d tweets' % len(tweets))
    pickle.dump(tweets, open('train_tweets.pkl', 'wb'))
    return tweets
    
def get_test_tweets(twitter,username):
    tweets = []
    for r in robust_request(twitter,'statuses/filter', {'track': username},5):
        tweets.append(r)
        if len(tweets) % 100 == 0:
            print('%d tweets' % len(tweets))
        if len(tweets) >= 500:
            break
    print('Retrieved %d tweets' % len(tweets))
    pickle.dump(tweets, open('test_tweets.pkl', 'wb'))
    return tweets

def get_friends(twitter, screen_name):
    users=defaultdict(dict)
    friends = robust_request(twitter,'friends/ids',{'screen_name':screen_name,'count':15},5)
    for friend in friends.json()["ids"]:
        users[friend]['names']=None
        users[friend]['friends']=[]
    pickle.dump(users, open('users.pkl', 'wb'))
    return users              

def get_friend_names(twitter, users):
    for u in users:
        response = robust_request(twitter,'users/lookup',{'user_id':u},5)
        users[u]['names']=response.json()[0]['screen_name']
    pickle.dump(users, open('users.pkl', 'wb'))
    return users
    
def get_friend_of_friends(twitter, users):
    for u in users:
        response = robust_request(twitter,'friends/ids',{'screen_name':users[u]["names"],'count':90},5)
        users[u]['friends']=response.json()["ids"]
    pickle.dump(users, open('users.pkl', 'wb'))
    return users
    
def log_collection_summary(train_tweets, test_tweets, users, user):
    f = open('collection_summary.txt', 'w')
    f.write("SUMMARY OF COLLECTION\n")
    f.write("training tweets count = "+str(len(train_tweets))+"\n")
    f.write("testing tweets count = "+str(len(test_tweets))+"\n")
    count=0
    for u in users:
        count = count + len(users[u]['friends'])
    f.write("User "+user+" 's friends are \n")
    for u in users:
       f.write("\t"+users[u]["names"]+"\n")
    f.write("users friend of friends' count = "+str(count)+"\n")

def main():
    twitter = get_twitter("API")
    print('Established Twitter connection.')
    user = "HillaryClinton"
    username = "Hillary Clinton"
    print("Getting training tweets")
    if os.path.isfile("train_tweets.pkl")==False:
        train_tweets = get_train_tweets(twitter, username)
    else:
        with open('train_tweets.pkl', 'rb') as f:
            train_tweets = pickle.load(f)
    print("Getting testing tweets")
    if os.path.isfile("test_tweets.pkl")==False:
        test_tweets = get_test_tweets(twitter, username)
    else:
        with open('test_tweets.pkl', 'rb') as f:
            test_tweets = pickle.load(f)
            
    if os.path.isfile("users.pkl")==False:
        users = get_friends(twitter,user)
        users = get_friend_names(twitter,users)
        users = get_friend_of_friends(twitter,users)
    else:
        with open('users.pkl', 'rb') as f:
            users = pickle.load(f)
    log_collection_summary(train_tweets, test_tweets, users, user)
    """
    #users = get_friends(twitter,'HillaryClinton')
    with open('users.pkl', 'rb') as f:
        users = pickle.load(f)
    #get_friend_names(twitter,users)
    get_friend_of_friends(twitter,users)
    print (users)
    #for u in users:
       #print(users[u]["names"])
    """
if __name__ == '__main__':
    main()
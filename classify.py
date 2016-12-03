"""
classify.py
"""
from collections import defaultdict
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import re
import pickle
import pandas as pd
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
import numpy as np

def AFINN_lexicon(afinn):
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')
    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])

    print('read %d AFINN terms.\nE.g.: %s' % (len(afinn), str(list(afinn.items())[:10])))
    
def afinn_sentiment2(terms, afinn, verbose=False):
    pos = 0
    neg = 0
    for t in terms:
        if t in afinn:
            if verbose:
                print('\t%s=%d' % (t, afinn[t]))
            if afinn[t] > 0:
                pos += afinn[t]
            else:
                neg += -1 * afinn[t]
    return pos, neg
    
def tokenize(text):
    return re.sub('\W+', ' ', text.lower()).split()
    
def analyze(tweets,afinn):
    tokens = []
    for t in tweets:
        if "text" in t.keys():
            tokens.append(tokenize(t['text']))
        
    for token_list, tweet in zip(tokens, tweets):
        pos, neg = afinn_sentiment2(token_list, afinn)
        if pos < neg:
            if "text" in tweet.keys():
                tweet['sentiment']=-1
        elif pos == neg:
            if "text" in tweet.keys():
                tweet['sentiment']=0
        elif pos > neg:
            if "text" in tweet.keys():
                tweet['sentiment']=1
    return tweets
                    
def vectorize(df, action, vocab):
    if action=="TRAIN":
        vectorizer = CountVectorizer(min_df=1, ngram_range=(1,1))
        X = vectorizer.fit_transform(df['text'])
        vocab=np.array(vectorizer.get_feature_names())
    else:
        vectorizer = CountVectorizer(min_df=1, ngram_range=(1,1), vocabulary=vocab)
        X = vectorizer.transform(df['text'])
        vocab=np.array(vectorizer.get_feature_names())
    return X, vocab
    
def accuracy(truth, predicted):
    return len(np.where(truth==predicted)[0]) / len(truth)
    
def do_cross_validation(model, X, y, n_folds):
    cv = KFold(len(y), n_folds)
    accuracies = []
    for train_ind, test_ind in cv:
        model.fit(X[train_ind], y[train_ind])
        predictions = model.predict(X[test_ind])
        accuracies.append(accuracy(y[test_ind], predictions))
    return model, accuracies

def p_dataframe(docs):
    arr=[]
    for d in docs:
        if "text" in d.keys():
            arr.append((d["text"], d["sentiment"]))
    labels = ["text", "sentiment"]
    df = pd.DataFrame.from_records(arr, columns=labels)
    return df
    
def log_classification_results(test_tweets, predictions, y2, accuracies):
    f = open('classification_summary.txt', 'w')
    f.write("SUMMARY OF CLASSIFICATION\n")
    f.write('Accuracy on training data = %.2f (std=%.2f)' %(np.mean(accuracies), np.std(accuracies))+"\n")
    f.write("Accuracy on testing data = "+str(accuracy(predictions,y2))+"\n")
    index=0
    f.write("Here is a set of tested tweets with a sentiment label (positive, negative, neutral)\n")
    for t in test_tweets:
        f.write(t["text"]+"\n")
        if predictions[index]==-1:
            f.write("Negative Tweet")
        elif predictions[index]==0:
            f.write("Neutral Tweet")
        elif predictions[index]==1:
            f.write("Positive Tweet")
        f.write("\n")
        index=index+1
        if index == 30:
            break   
            
def main():
    afinn = dict()
    AFINN_lexicon(afinn)
    with open('train_tweets.pkl', 'rb') as f:
        train_tweets = pickle.load(f)
    with open('test_tweets.pkl', 'rb') as f:
        test_tweets = pickle.load(f)
    train_t = analyze(train_tweets, afinn)
    test_t = analyze(test_tweets,afinn)
    df = p_dataframe(train_t)
    tf = p_dataframe(test_t)
    vocab=None
    X1, vocab = vectorize(df,"TRAIN",vocab)
    y1 = np.array(df["sentiment"])
    model, accuracies = do_cross_validation(LogisticRegression(), X1, y1, 5)
    X2, vocab = vectorize(tf,"TEST", vocab)
    y2 = np.array(tf["sentiment"])
    predictions = model.predict(X2)
    log_classification_results(test_tweets, predictions, y2, accuracies)
    
if __name__ == '__main__':
    main()
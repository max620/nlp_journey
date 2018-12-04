# -*- coding: utf-8 -*-
"""
@author: Max
"""
import os
import json
import csv
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

folder_path = r"C:\Users\Max\Anaconda3\Dataset\Text Mining Assignment"
data =  open(os.path.join(folder_path, "Preprocessed_text_Max.csv"), "r")

df = pd.read_csv(r'C:\Users\Max\Anaconda3\Dataset\Text Mining Assignment\Preprocessed_text_Max.csv', encoding="utf8", sep=",")
df.head(3)
df.columns
df1 = df.drop(['porter','lancaster','snowball'], axis=1)
df1.columns
df1.head(3)
sorted(df1)

df2 = df1.join(df1['review_user_loc'].str.split(',',2, expand=True).rename(columns={0:'City', 1:'Country'}))
df2.head(3)
df2.columns
type(df2)

###freq date

fd = nltk.FreqDist(df2['review_date'])
fd.most_common(10)
fd.plot(10, cumulative=False)

###by most popular date
df3_2017 = df.query('review_date == "11 July 2017"|review_date == "11 April 2017"|review_date == "3 October 2017"|review_date == "14 February 2017"|review_date == "6 September 2017"|review_date == "14 March 2017"')
df3_2018 = df.query('review_date == "19 June 2018"|review_date == "10 July 2018"|review_date == "7 August 2018"|review_date == "4 March 2018"')
df3_date = df.query('review_date == "11 July 2017"|review_date == "11 April 2017"|review_date == "3 October 2017"|review_date == "14 February 2017"|review_date == "6 September 2017"|review_date == "14 March 2017"|review_date == "19 June 2018"|review_date == "10 July 2018"|review_date == "7 August 2018"|review_date == "4 March 2018"')

def df_flower(df_name):
    global df_flower_data
    df_flower_data = pd.DataFrame(df_name.query('attraction_name == "Flower Dome"'))
    return df_flower_data
 
def df_cloud(df_name):
    global df_cloud_data
    df_cloud_data = pd.DataFrame(df_name.query('attraction_name == "Cloud Forest"'))
    return df_cloud_data
   
def df_garden(df_name):
    global df_garden_data
    df_garden_data = pd.DataFrame(df_name.query('attraction_name == "Gardens by the Bay"'))
    return df_garden_data

def df_gc(df_name):
    global df_gc_data
    df_gc_data = pd.DataFrame(df_name.query('attraction_name == "Flower Dome"|attraction_name == "Cloud Forest"'))
    return df_gc_data
    
###wordCloud
def wc(df_name):
    df_text = pd.DataFrame(df_name, columns=['lemmatization'])
    df_text = pd.DataFrame.to_string(df_text) 
    wc = WordCloud(background_color="white").generate(df_text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()

df_flower(df3_2017)
wc(df_flower_data)
df_flower(df3_2018)
wc(df_flower_data)

df_cloud(df3_2017)
wc(df_cloud_data)
df_cloud(df3_2018)
wc(df_cloud_data)

df_garden(df3_2017)
wc(df_garden_data)
df_garden(df3_2018)
wc(df_garden_data) #one day visit

### collocation
def bigram(collat_data):
    df_co = pd.DataFrame.to_string(collat_data, columns=['lemmatization']).split(',')
    bcf = BigramCollocationFinder.from_words(df_co)
    top20 = bcf.nbest(BigramAssocMeasures.likelihood_ratio, 20)
    return top20

df_flower(df3_date)
df_cloud(df3_date)
df_garden(df3_date)

bigram(df_flower_data)
bigram(df_cloud_data)
bigram(df_garden_data)

def trigram(collat_data):
    df_co = pd.DataFrame.to_string(collat_data, columns=['lemmatization']).split(',')
    tcf = TrigramCollocationFinder.from_words(df_co)
    top20 = tcf.nbest(TrigramAssocMeasures.likelihood_ratio, 20)
    return top20

trigram(df_flower_data)
trigram(df_cloud_data)
trigram(df_garden_data)

#bcf.nbest(BigramAssocMeasures.chi_sq, 10)
#bcf.nbest(BigramAssocMeasures.pmi, 10)
#bcf.nbest(BigramAssocMeasures.raw_freq, 10)

### tf-idf vectorizer with pos-weighted 
from sklearn.feature_extraction.text import TfidfVectorizer

pos_garden = pd.DataFrame.to_string(df3_date, columns=['lemmatization']).split(',')

vec_tfidf = TfidfVectorizer(stop_words="english", min_df= 2, sublinear_tf=True, use_idf=True)
garden_tfidf = vec_tfidf.fit_transform(pos_garden)
garden_tfidf.shape

indices = np.argsort(vec_tfidf.idf_) 
features = vec_tfidf.get_feature_names()
top_n = 20
top_features = [features[i] for i in indices[:top_n]]
top_features

pos_garden_tag = pos_tag(top_features)
pos_garden_tag
sent_chunk = ne_chunk(pos_garden_tag)
print(sent_chunk)

#meaning of POS tag
import nltk
nltk.app.chunkparser_app.app()

### supervised learning
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2

df3_date.shape
df_gc(df3_date)
df_gc_data.columns

cluster_garden = pd.DataFrame(df3_date, columns=['attraction_name','lemmatization'])
cluster_garden.shape

x_train, x_test, y_train, y_test = train_test_split(cluster_garden['lemmatization'], cluster_garden['attraction_name'], test_size=0.2)

# Model 1
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words="english", min_df= 2, sublinear_tf=True, use_idf=True)),
                     ('chi', SelectKBest(chi2, k=1000)),
                     ('clf', MultinomialNB())
                     ])

model = pipeline.fit(x_train,y_train)

tfidf = model.named_steps['tfidf']
chi = model.named_steps['chi']  
clf = model.named_steps['clf']  

feature_names = tfidf.get_feature_names()
feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
feature_names = np.asarray(feature_names)
feature_names

predict = model.predict(x_test) #to predict class values

print("\nModel 1")
print("Accuracy Score: " + str(model.score(x_test, y_test)))
print('Review is:', predict[0:2])

# Model 2 # Accuracy highest at 0.79
from sklearn.svm import LinearSVC

pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words="english", min_df= 2, sublinear_tf=True, use_idf=True)),
                     ('chi', SelectKBest(chi2, k=1000)),
                     ('clf', LinearSVC(penalty='l1', max_iter=5000, dual=False))
                     ])

model = pipeline.fit(x_train,y_train)

tfidf = model.named_steps['tfidf']
chi = model.named_steps['chi']  
clf = model.named_steps['clf']  

feature_names = tfidf.get_feature_names()
feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
feature_names = np.asarray(feature_names)
feature_names

predict = model.predict(x_test) #to predict class values

print("\nModel 2")
print("Accuracy Score: " + str(model.score(x_test, y_test)))
print('Review is:', predict[0:2])

# Model 3
from sklearn.linear_model import SGDClassifier

pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words="english", min_df= 2, sublinear_tf=True, use_idf=True)),
                     ('chi', SelectKBest(chi2, k=1000)),
                     ('clf', SGDClassifier(penalty='l2', max_iter=5000))
                     ])

model = pipeline.fit(x_train,y_train)

tfidf = model.named_steps['tfidf']
chi = model.named_steps['chi']  
clf = model.named_steps['clf']  

feature_names = tfidf.get_feature_names()
feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
feature_names = np.asarray(feature_names)
feature_names

predict = model.predict(x_test) #to predict class values

print("\nModel 3")
print("Accuracy Score: " + str(model.score(x_test, y_test)))
print('Review is:', predict[0:2])

# Model 4
from sklearn import tree

pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words="english", min_df= 2, sublinear_tf=True, use_idf=True)),
                     ('chi', SelectKBest(chi2, k=1000)),
                     ('clf', tree.DecisionTreeClassifier())
                     ])

model = pipeline.fit(x_train,y_train)

tfidf = model.named_steps['tfidf']
chi = model.named_steps['chi']  
clf = model.named_steps['clf']  

feature_names = tfidf.get_feature_names()
feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
feature_names = np.asarray(feature_names)
feature_names

predict = model.predict(x_test) #to predict class values

print("\nModel 4")
print("Accuracy Score: " + str(model.score(x_test, y_test)))
print('Review is:', predict[0:2])

############################## NOT USING BELOW ############################
### date count by attraction_name and country
array = ['Gardens by the Bay','Cloud Forest','Flower Dome']
df2 = df2[df2['attraction_name'].isin(array)]
df2.head(3)

array[0] #Gardens by the Bay
array[1] #Cloud Forest
array[2] #Flower Dome

df_summary = df2[df2['attraction_name'].str.contains(array[0])].groupby(['attraction_name','Country','review_date']).count().sort_values(by=['lemmatization'],ascending=False)
df_summary.head(5)

df_summary2 = df2[df2['attraction_name'].str.contains(array[1])].groupby(['attraction_name','Country','review_date']).count().sort_values(by=['lemmatization'],ascending=False)
df_summary2.head(5)

df_summary3 = df2[df2['attraction_name'].str.contains(array[2])].groupby(['attraction_name','Country','review_date']).count().sort_values(by=['lemmatization'],ascending=False)
df_summary3.head(5)

df_summary['lemmatization']

#with json file
read = csv.DictReader(df, fieldnames = "Index")
df_json = json.dumps([row for row in read])
f = open(os.path.join(folder_path, "parsed.json"), 'w') 
f.write(df_json)
print(df_json)

len(df_json)
type(df_json)
token = word_tokenize(df_json)
len(token)
unique = set(token)
len(token)/len(unique)
sorted(unique)
single = [w for w in unique if len(w) >= 1 ]
len(single)
single

#with csv file direct
lem = WordNetLemmatizer()
stem = PorterStemmer()
eng_stopwords = stopwords.words('english')
a = df.to_string()
fg = stem.stem(a)
wordList = word_tokenize(fg)                                     
wordList = [word for word in wordList if word not in eng_stopwords] 
print (wordList)
len(wordList)
type(wordList)
unique_fg = set(wordList)
sorted(unique_fg)
single_fg = [w for w in unique if len(w) >= 1 ]
len(single_fg)






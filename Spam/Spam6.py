import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('spam.tsv', sep="\t")
#print(df.head())
print(df)

hamDf = df[df['label']=="ham"]
spamDf = df[df['label']=="spam"]
#print(hamDf.head())
#print(spamDf.head())

print(hamDf.shape)
print(spamDf.shape)


hamDf = hamDf.sample(spamDf.shape[0]) #ham msg are more than spam so sample 
#sample is used to randomly collect data from ham which are equal to 747

#print(hamDf.shape)
#print(spamDf.shape)

#combine both dataset spamdf and hamdf
finalDf = pd.concat([hamDf, spamDf], ignore_index=True)
#print(finalDf.shape)

X_train, X_test, Y_train, Y_test = train_test_split(
    finalDf['message'], finalDf['label'], test_size=0.2, random_state=0, shuffle=True, stratify= finalDf['label'])

#classfication of ham and spam
#we use pipeline for classification which helps to keep machine busy in worki

#model = Pipeline([('tfidf', TfidfVectorizer()), ('model', RandomForestClassifier(n_estimators=100, n_jobs=-1))])

#n_estimators -->no of trees in forest
#n_jobs --> no of jobs run in parallel
#TfidfTransformer--> tfidf vector--> frequency matrix are created according to the repeatation
# it means how many time a word is repeating in sentence
#Tf = no of rep of words in sentence/no of words in sentence

#model = Pipeline([('tfidf', TfidfVectorizer()), ('model', SVC(C=1000, gamma='auto'))])
#model = Pipeline([('tfidf', TfidfVectorizer()), ('model', DecisionTreeClassifier(max_depth=5))])
model = Pipeline([('tfidf', TfidfVectorizer()), ('model', MultinomialNB())])


model.fit(X_train, Y_train)

Y_predict = model.predict(X_test)

#print(confusion_matrix(Y_test, Y_predict))
#print(classification_report(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict))

#print(model.predict(["you have won a $500 gift card to Target. Click here to claim your reward"]))

#joblib.dump(model, "RandomForest.pkl")
#joblib.dump(model, "SpportVectorMachine.pkl")
#joblib.dump(model, "DecisionTreeClassifier.pkl")
#joblib.dump(model, "NaiveBayes.pkl")

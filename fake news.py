import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import model_selection

data = pd.read_csv("/Users/amanpuranik/Desktop/fake-news-detection/data.csv")
data = data[['Headline', "Label"]]

'''x = data["Headline"]
y = data["Label"]'''

training = np.array(data["Headline"])
testing = np.array(data["Label"])





x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
#training, testing = model_selection.train_test_split(data, test_size=0.2)
print(len(training))
print(len(testing))
model = MultinomialNB()

model.fit(training, testing)






tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#model.fit(x_train,y_train) #this part gives me a string to float error

pipeline = Pipeline([('vectorizer', tfidf_vectorizer), ('classifier', model)])
pipeline.fit(x_train, y_train)
accuracy = pipeline.score(x_test, y_test)
a = float(x_test)


predictions = model.predict(a)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])






'''tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)'''


import numpy as np
import pandas as pd
import nltk
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from stemming.porter2 import stem
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
import pickle
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap

#nltk.download()


#reading in the data
data = pd.read_csv("/Users/amanpuranik/Desktop/fake-news-detection/data.csv")
data = data[['Headline', "Label"]]

x = np.array(data['Headline'])
y = np.array(data["Label"])

# tokenization of the data here'
headline_vector = []

for  headline in x:
    headline_vector.append(word_tokenize(headline))

#print(headline_vector)

stopwords = set(stopwords.words('english'))

#removing stopwords at this part
filtered = [[word for word in sentence if word not in stopwords]
            for sentence in headline_vector]
#print(filtered)

#stemming the headlines
ps = PorterStemmer()

stemmed2 = [[stem(word) for word in headline] for headline in filtered]
#print(stemmed2)

#lowercase
lower = [[word.lower() for word in headline] for headline in stemmed2] #start here

#conver lower into a list of strings
lower_sentences = [" ".join(x) for x in lower]
#print(lower_sentences)


#part of speech
pos = [nltk.pos_tag(word) for word in lower]

#lemmatising the speech
lem = WordNetLemmatizer()

lem_words = [[lem.lemmatize(word) for word in headline] for headline in filtered]
#print(lem_words)


#organising
articles = []


for headline in lower:
    articles.append(headline)

#print(articles[0])


#frequency
all_words = []
for headline in lower:
    all_words.append(nltk.FreqDist(headline))
#print(all_words)


#CREATING BAG OF WRODS MODEL
headline_bow = CountVectorizer()
b = headline_bow.fit(lower_sentences)
#print(headline_bow.get_feature_names())
dictionary = headline_bow.get_feature_names()

#print(dictionary) #i think this part is the actual part where its making the dictionary of words
#print(dictionary[442])
a = headline_bow.transform(lower_sentences) #here is the bag of words
#print(b.vocabulary_)
#print(a.shape)
xxx = a.toarray()
#print(xxx) #this is where it prints out the vectorised versions of each headline
#print(a) #this prints out the actual bag of wrods


#testing and training part
yy = np.reshape(y,(-1,1))
x_train, x_test, y_train, y_test = train_test_split(a, yy, test_size=0.2, random_state=1)

'''print(a[0])
print(yy[0])
'''


#fitting on the model now

model = MultinomialNB() #don forget these brackets here
model.fit(x_train,y_train.ravel())

#print(x_test[6], y_test[6])

accuracy = model.score(x_test,y_test)
#print(accuracy)


#ANOTHER MODEL (SVC)
model2 = SVC()
model2.fit(x_train,y_train)
print(model2.score(x_test,y_test))


#LINEAR MODEL
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
print(linear.score(x_test, y_test)) #as expected, linear model has the lowest score


#CONFUSION MATRIX
y_pred = model.predict(x_test)
#print(y_pred) #i can iterate through this
confusion = confusion_matrix(y_test,y_pred)
print(confusion)


#TRYING TO USE NEW DATA (NEW HEADLINES THAT WERE NOT PART OF THE DATASET
new_data = []
test = ['est', "test", "test"]

input = input('Type headline here')
new_data.append(word_tokenize(input))
#print(new_data)

new_data_stop = [[word for word in sentence if word not in stopwords]for sentence in new_data]
#print(new_data_stop)

new_data_stem = [[stem(word) for word in headline] for headline in new_data_stop]

new_data_lower = [[word.lower() for word in headline] for headline in new_data_stem]
#print(new_data_lower)

new_data2 = [" ".join(x) for x in new_data_lower]

#new_vecto = headline_bow.fit(new_data2)
new_vector = headline_bow.transform(new_data2)

print(new_vector)

for list in new_data:
    for word in list:
        test.append(word) #this is what I wanted





#MAKING PREDICTIONS BASED ON NEW DATA
predict = model2.predict(new_vector)
print(predict)




#MAKING THE FRONTEND

app = Flask(__name__)
Bootstrap(app)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # get the form data
        text = request.form['input']
        # do your prediction here
        render_template('index.html', text=text)
    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug = True)


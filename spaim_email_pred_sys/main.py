from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Reading CSV file to the Pandas Dataframe
spam_df = pd.read_csv("spam.csv")
# print(spam_df) #print Pandas Data Frame

# applying groupby() function to
# group the data on Category value.
spam_df.groupby('Category').describe()

#Creating new colom 'spam'
spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
#Dividing the Data into Training and Testing Data
x_train,x_test,y_train,y_test = train_test_split(spam_df.Message,spam_df.spam, test_size = 0.25)
#extract numerical features from text content
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values)
#Training of Model
model = MultinomialNB()
model.fit(x_train_count, y_train)
x_test_count = cv.transform(x_test)

"""
#Finding Accuracy of the Model
accuracy = model.score(x_test_count,y_test)
print('Accuracy of the model is : ', accuracy)
"""

y_predic = model.predict(x_test_count)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['EMAIL']
        data = [message]
        email_haim_count = cv.transform(data)
        haim_spam = model.predict(email_haim_count)
        if haim_spam == 0:
            email_status = 'Email is Haim'
            return render_template('index.html', prediction=email_status)
        else:
            email_status = 'Looking Spam be Caire full'
            return render_template('index.html', prediction=email_status)


if __name__ == '__main__':
    app.run(debug=True)


"""
#Creating confusion Matrix
cm = confusion_matrix(y_test, y_predic)
print("The confusion Matrix is: ")
print(cm)

plt.figure(figsize=(7,7))
sns.heatmap(cm, annot=True)
plt.xlabel = ('predicted')
plt.ylabel = ('Truth')
plt.show()
"""

"""
# For free Test
email_ham = []
list1 = input('Inter Your Email : ')
email_ham.append(list1)
email_haim_count = cv.transform(email_ham)
pre_test = model.predict(email_haim_count)
if pre_test == [0]:
    print("Email is harm.")
else:
    print("Email is spam.")
"""


"""
email_spa = ['reward mony click']
email_spa_count = cv.transform(email_spa)
pre_test = model.predict(email_spa_count)
print(pre_test)
"""





"""
app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/about")
def talha():
    name = "talha"
    return render_template('index.html', name2=name)
    # return "hellow"

app.run(debug=True)
"""
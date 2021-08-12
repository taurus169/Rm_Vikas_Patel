# Detecting Hate Speech on Twitter
<p>Detection of hate speeches on twitter can be done with the help of Machine Learning and Natural Language Processing. ML and NLP algorithms are based on statistical methods they are fed training data and from that data model learns repetitively and at last the testing data is tested on the model that is learned. Accuracy may vary on basis of different algorithms and datasets used.</p>
<p>Here we have used three different algorithms for detecting hate speech on twitter. It is found that Support Vector Machine gives an accuracy score of 0.944 with the dataset used here. Here the data is split 70:30 for training and testing respectively. It's worth noting that Logistic Regression produced excellent results as well with an accuracy score of 0.940.</p>
<h2> Dataset: </h2>
<p>Hate-speech detection was studied using a dataset based on Twitter data. Hate-speech, offensive language, and neither are the classifications for the data. It's crucial to highlight that this dataset contains language that might be deemed racist, sexist, homophobic, or otherwise objectionable due to the nature of the study. </p>
<p><b>Link for dataset: https://www.kaggle.com/mrmorj/hate-speech-and-offensive-language-dataset</b></p>
<h2> Here we have used three machine learning algorithms:</h2>
<ul>
<li><p><b>Naive Bayes:</b> Naive Bayes are a group of supervised machine learning classification algorithms based on the Bayes theorem. It's a basic classification method with high functionality. Gaussian Naive Bayes is a Naive Bayes version that uses a Gaussian normal distribution and can handle continuous data.</p></li>
<li><p><b>Support Vector Machine:</b> Support Vector Machine is a supervised machine learning method that can be used for classification and regression. Support Vector Machines are based on the concept of determining the optimal hyperplane for dividing a dataset into two groups. ![Svm](https://github.com/taurus169/Rm_Vikas_Patel/blob/main/Images/Svm.png)</p></li>
<li><p><b>Logistic Regression:</b> Logistic Regression is a predictive method that uses independent factors to predict the dependent variable, where the dependent variable must be a categorical variable. It assumes a linear relationship between the input variables with the output.</p></li>
</ul>
<h2> Tools used for project development: </h2>
<ul>
<li><p><b>Python</b></p></li>
<li><p><b>NLP</b></p></li>
<li><p><b>Porter Stemmer</b></p></li>
<li><p><b>Count Vectorizer</b></p></li>
<li><p><b>Support Vector Machine</b></p></li>
<li><p><b>Logistic Regression</b></p></li>
<li><p><b>Gaussian Naive Bayes Classifier</b></p></li>
</ul>
<h2>Running Code and Usage</h2>

- This project is done using python programming. It is best to use python3(latest python version). I have run this code on google colab, you can do the same. First of all you need to download the dataset from https://www.kaggle.com/mrmorj/hate-speech-and-offensive-language-dataset . After downloading the dataset, open a new notebook on google colab and give it a name. You can upload the dataset csv file using drag and drop to folders part on left side of the page. Python 3 environment comes with many helpful analytics libraries installed.
-  Importing libraries and packages.
```bash
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```
- Importing the dataset

```bash
dataset = pd.read_csv('/content/labeled_data.csv')
dataset.head()
```
- Encoding the dependent variable
```bash
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
y = np.array(ct.fit_transform(y))
```
  This data has been split into two variables that will be used to fit hate speech and offensive speech models.
```bash
y_df = pd.DataFrame(y)
y_hate = np.array(y_df[0])
y_offensive = np.array(y_df[1])
```
- Cleaning the Texts

Performing  PorterStemmer

```bash
corpus = []
for i in range(0, 24783):
  review = re.sub('[^a-zA-Z]', ' ', dt_trasformed['tweet'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)
 ```
 
Performing CountVectorizer

```bash
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(corpus).toarray()
```  
- Splitting the dataset into the Training set and Test set

```bash
X_train, X_test, y_train, y_test = train_test_split(X, y_hate, test_size = 0.30, random_state = 0)
```
-Finding the best models to predict hate speech

NaiveBayes

```bash
classifier_np = GaussianNB()
classifier_np.fit(X_train, y_train)
```

Logistic Regression

```bash
classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(X_train, y_train)
```

Support Vector Machine

```bash
classifier_svm = svm.SVC()
classifier_svm.fit(X_train, y_train)
```

- Making the Confusion Matrix for each model

NaiveBayes

```bash
y_pred_np = classifier_np.predict(X_test)
cm = confusion_matrix(y_test, y_pred_np)
print(cm)
```
Support Vector Machine

```bash
y_pred_svm = classifier_svm.predict(X_test)
cm = confusion_matrix(y_test, y_pred_svm)
print(cm)
```
Logistic Regression

```bash
y_pred_lr=classifier_lr.predict(X_test)
cm = confusion_matrix(y_test, y_pred_lr)
print(cm)
```
Accuracy Score for all three models

```bash
svm_score = accuracy_score(y_test, y_pred_svm)
lr_score = accuracy_score(y_test, y_pred_lr)
np_score = accuracy_score(y_test, y_pred_np)

print('Support Vector Machine Accuracy: ', str(svm_score))
print('Logistic Regression Accuracy: ',str(lr_score))
print('Naive Bayes Accuracy: ', str(np_score))
```
Support Vector Machine Accuracy:  0.9440484196368527<br>
Logistic Regression Accuracy:  0.9402824478816408<br>
Naive Bayes Accuracy:  0.4772024209818426

<p> So Based on this dataset, Support Vector Machine appears to be a superior predictor of hate speech. It's worth noting that Logistic Regression produced excellent results as well. This Dataset appears to be an artificial intelligence product used to classify hate and abusive speech.</p>

<h2>Support</h2>

- See the [full project overview](https://github.com/taurus169/Rm_Vikas_Patel/blob/main/RM_Project_hate_speech_detection.ipynb) in the `RM_Project_hate_speech_detection.ipynb` Jupyter Notebook.

- For additional information or suggestions, contact Vikas Patel at [paarav98@gmail.com](mailto:paarav98@gmail.com)

**Let's connect!**

<a href="https://www.linkedin.com/in/taurus169/" target="_blank"><img alt="LinkedIn" src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" /></a>

<h2>Authors and Acknowledgements</h2>

<h2>Next Steps</h2>

To further develop this project, here are some immediate next steps that anyone could execute.

- Collect more potential "Hate Speech" data
- Improve final model with different preprocessing techniques, such as removing offensive language as stop words
- Evaluate model with new tweets or other online forum data to see if it can generalize well
- Applying Optimizing Techniques

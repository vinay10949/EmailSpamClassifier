import matplotlib.pyplot as plot #for plotting histograms
import csv
from textblob import TextBlob
import pandas #to the read the file
import cPickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer #For tokenizing the terms and building TF-IDF
from sklearn.naive_bayes import MultinomialNB #use of naivebayes
from sklearn.svm import SVC #use of SVM for classfication
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.learning_curve import learning_curve
#import nltk
#from nltk.corpus import stopwords

#import the emails using pandas   
emails = pandas.read_csv('/home/vinay10949/SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE, names=["label", "message"])

#print emails.groupby('label').describe() #Display the statistics of the message

emails['length'] = emails['message'].map(lambda text: len(text)) #Store length in the emails

plot.hist(emails['length'],bins=30,label="Number of sms of varying length") #print histogram

emails.length.describe() #PRtint statistics of SMS

#print list(emails.message[emails.length > 900]) #Print the longest message

emails.hist(column='length', by='label', bins=50) # Print the histogram for emails by label (spam,ham)


#Data Preprocessing
def tokenize(message):
    message = unicode(message, 'utf8')  # convert bytes into proper unicode
    return TextBlob(message).words
    
def lemmatize(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]


emails_trainsform = CountVectorizer(analyzer=lemmatize).fit(emails['message'])
#print len(emails_trainsform.vocabulary_)


print list(emails.message[emails.length > 900]) #Print the longest message

emails.hist(column='length', by='label', bins=50) # Print the histogram for emails by label (spam,ham)


emails_bow = emails_trainsform.transform(emails['message'])
print 'sparse matrix shape:', emails_bow.shape
print 'number of non-zeros:', emails_bow.nnz
print 'sparsity: %.2f%%' % (100.0 * emails_bow.nnz / (emails_bow.shape[0] * emails_bow.shape[1]))


tfidf_transformer = TfidfTransformer().fit(emails_bow)

emails_tfidf = tfidf_transformer.transform(emails_bow) #Transforming bag of words into corpus
print emails_tfidf.shape

spam_detector = MultinomialNB().fit(emails_tfidf, emails['label'])

all_predictions = spam_detector.predict(emails_tfidf)
print all_predictions

print 'Accuray', accuracy_score(emails['label'], all_predictions)
print 'Confusion matrix\n', confusion_matrix(emails['label'], all_predictions)
print '(row=expected, col=predicted)'


plot.matshow(confusion_matrix(emails['label'], all_predictions), cmap=plot.cm.binary, interpolation='nearest')
plot.title('confusion matrix')
plot.colorbar()
plot.ylabel('expected label')
plot.xlabel('predicted label')
print classification_report(emails['label'], all_predictions)


#Dividing data set
msg_train, msg_test, label_train, label_test = \
    train_test_split(emails['message'], emails['label'], test_size=0.2)

print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)



pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=lemmatize)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


scores = cross_val_score(pipeline,  # steps to convert raw emails into models
                         msg_train,  # training data
                         label_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )
print scores


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plot.figure()
    plot.title(title)
    if ylim is not None:
        plot.ylim(*ylim)
    plot.xlabel("Training examples")
    plot.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plot.grid()

    plot.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plot.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plot.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plot.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plot.legend(loc="best")
    return plot
plot_learning_curve(pipeline, "Accuracy vs. Training set size", msg_train, label_train, cv=5)

print scores.mean(), scores.std()

params = {
    'tfidf__use_idf': (True, False),
    'bow__analyzer': (lemmatize, tokenize),
}

grid = GridSearchCV(
    pipeline,  # pipeline from above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best found param combination
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)

nb_detector = grid.fit(msg_train, label_train)
print nb_detector.grid_scores_


print nb_detector.predict_proba(emails['message'][2])[0]

print nb_detector.predict(emails['message'][2])[0]

predictions = nb_detector.predict(msg_test)
print confusion_matrix(label_test, predictions)
print classification_report(label_test, predictions)


#SVM Classifier
pipeline_svm = Pipeline([
    ('bow', CountVectorizer(analyzer=lemmatize)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),  # <== change here
])

# pipeline parameters to automatically explore and tune
param_svm = [
  {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
  {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]


grid_svm = GridSearchCV(
    pipeline_svm,  # pipeline from above
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # Crossvalidation tyme
)

svm_detector = grid_svm.fit(msg_train, label_train) # find the best combination from param_svm
print svm_detector.grid_scores_

print  svm_detector.predict(emails['message'][2])[0]

print confusion_matrix(label_test, svm_detector.predict(msg_test))
print classification_report(label_test, svm_detector.predict(msg_test))

# store the spam detector to disk after training
with open('ClassifierModel.pkl', 'wb') as fout:
    cPickle.dump(svm_detector, fout)

#Loading the model on to different machine
svm_detector_reloaded = cPickle.load(open('ClassifierModel.pkl'))

print 'before:', svm_detector.predict([emails])[0]
print 'after:', svm_detector_reloaded.predict([emails])[0]


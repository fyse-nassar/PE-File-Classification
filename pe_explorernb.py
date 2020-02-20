
import sklearn.feature_extraction
import pandas as pd
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib.pyplot as plt

# %matplotlib inline
plt.rcParams['font.size'] = 18.0
plt.rcParams['figure.figsize'] = 16.0, 5.0


def plot_cm(cm, labels):
    # Compute percentages
    percent = (cm * 100.0) / np.array(np.matrix(cm.sum(axis=1)).T)  # Derp, I'm sure there's a better way
    print 'Confusion Matrix Stats'
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            print "%s/%s: %.2f%% (%d/%d)" % (label_i, label_j, (percent[i][j]), cm[i][j], cm[i].sum())

    # Show confusion matrix
    # Thanks kermit666 from stackoverflow :)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(b=False)
    cax = ax.matshow(percent, cmap='coolwarm', vmin=0, vmax=100)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


import pe_features

my_extractor = pe_features.PEFileFeatures()

# Open a PE File and see what features we get
filename = 'data/test.exe'
with open(filename, 'rb') as f:
    features = my_extractor.execute(f.read())
features


# Load up all our files (files come from various places contagio, around the net...)
def load_files(file_list):
    features_list = []
    for filename in file_list:
        with open(filename, 'rb') as f:
            features_list.append(my_extractor.execute(f.read()))
    return features_list


# Family 1 files
file_list = [os.path.join('data/', child) for child in os.listdir('data/')]
family1_features = load_files(file_list)
print 'Loaded up %d Family 1 PE Files' % len(family1_features)

# Family 2 files
file_list = [os.path.join('data/', child) for child in os.listdir('data/')]
family2_features = load_files(file_list)
print 'Loaded up %d Family 2 PE Files' % len(family2_features)

# Putting the features into a pandas dataframe

df_fam1 = pd.DataFrame.from_records(family1_features)
df_fam1['label'] = 'fam1'
df_fam2 = pd.DataFrame.from_records(family2_features)
df_fam2['label'] = 'fam2'
df_fam2.head()

# Now we're set and we open up a a whole new world!

# Gisting and statistics
df_fam1.describe()

# Visualization I
df_fam1['check_sum'].hist(alpha=.5, label='fam1', bins=40)
df_fam2['check_sum'].hist(alpha=.5, label='fam2', bins=40)
plt.legend()

# Visualization I
df_fam1['generated_check_sum'].hist(alpha=.5, label='fam1', bins=40)
df_fam2['generated_check_sum'].hist(alpha=.5, label='fam2', bins=40)
plt.legend()

# Concatenate the info into a big pile!
df = pd.concat([df_fam1, df_fam2], ignore_index=True)
df.replace(np.nan, 0, inplace=True)

# Boxplots show you the distribution of the data (spread).
# http://en.wikipedia.org/wiki/Box_plot

# Get some quick summary stats and plot it!
df.boxplot('number_of_import_symbols', 'label')
plt.xlabel('fam1 vs. fam2 files')
plt.ylabel('# Import Symbols')
plt.title('Comparision of # Import Symbols')
plt.suptitle("")

# Get some quick summary stats and plot it!
df.boxplot('number_of_sections', 'label')
plt.xlabel('fam1 vs. fam2 files')
plt.ylabel('Num Sections')
plt.title('Comparision of Number of Sections')
plt.suptitle("")

# Split the classes up so we can set colors, size, labels
cond = df['label'] == 'fam2'
fam2 = df[cond]
fam1 = df[~cond]
plt.scatter(fam2['number_of_import_symbols'], fam2['number_of_sections'],
            s=140, c='#aaaaff', label='fam2', alpha=.4)
plt.scatter(fam1['number_of_import_symbols'], fam1['number_of_sections'],
            s=40, c='r', label='fam1', alpha=.5)
plt.legend()
plt.xlabel('Import Symbols')
plt.ylabel('Num Sections')

# In preparation for using scikit learn we're just going to use
# some handles that help take us from pandas land to scikit land

# List of feature vectors (scikit learn uses 'X' for the matrix of feature vectors)
X = df.as_matrix(['number_of_import_symbols', 'number_of_sections'])

# Labels (scikit learn uses 'y' for classification labels)
y = np.array(df['label'].tolist())

# Random Forest is a popular ensemble machine learning classifier.
# http://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestClassifier.html

import sklearn.ensemble

# TODO: Check compute_importance
#clf = sklearn.ensemble.RandomForestClassifier(n_estimators=50, compute_importances=True)
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=50)
from sklearn.model_selection import cross_val_score
# Now we can use scikit learn's cross validation to assess predictive performance.
scores = cross_val_score(clf, X, y, cv=2, n_jobs=4)
print scores

# Typically you train/test on an 80% / 20%  split meaning you train on 80%
# of the data and you test against the remaining 20%. In the case of this
# exercise we have so FEW samples (50 good/50 bad) that if were going
# to play around with predictive performance it's more meaningful
# to train on 60% of the data and test against the remaining 40%.

my_seed = 123
my_tsize = .4  # 40%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=my_tsize, random_state=my_seed)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Now plot the results of the 60/40 split in a confusion matrix
from sklearn.metrics import confusion_matrix

labels = ['fam2', 'fam1']
cm = confusion_matrix(y_test, y_pred, labels)
plot_cm(cm, labels)

# Okay now try putting in ALL the features (except the label, which would be cheating :)
no_label = list(df.columns.values)
no_label.remove('label')
X = df.as_matrix(no_label)

# 60/40 Split for predictive test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=my_tsize, random_state=my_seed)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels)
plot_cm(cm, labels)

# Feature Selection
# Which features best deferentiated the two classes?
# Here we're going to grab the feature_importances from the classifier itself,
# you can also use a Chi Squared Test sklearn.feature_selection.SelectKBest(chi2)
importances = zip(no_label, clf.feature_importances_)
importances.sort(key=lambda k: k[1], reverse=True)
importances[:10]

# Produce an X matrix with only the most important featuers
X = df.as_matrix([item[0] for item in importances[:10]])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=my_tsize, random_state=my_seed)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels)
plot_cm(cm, labels)

# Compute the predition probabilities and use them to mimimize our false positives
# Note: This is simply a trade off, it means we'll miss a few of the malicious
# ones but typically false alarms are a death blow to any new 'fancy stuff' so
# we definitely want to mimimize the false alarms.
y_probs = clf.predict_proba(X_test)[:, 0]
thres = .8  # This can be set to whatever you'd like
y_pred[y_probs < thres] = 'fam2'
y_pred[y_probs >= thres] = 'fam1'
cm = confusion_matrix(y_test, y_pred, labels)
plot_cm(cm, labels)

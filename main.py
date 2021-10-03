import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
import nltk
import re
import pickle
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
stop = stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer


pd.set_option('display.max_colwidth', None)

racistText = pd.read_csv("annotations.csv")
racistText.head()
sexistText = pd.read_csv("labeled_data.csv")
sexistText.head(100)
positiveNegativeNeutralText = pd.read_csv("positiveNeutral.csv",encoding='latin1')
positiveNegativeNeutralText.columns = ["target", "id", "date", "flag", "user", "text"]

positiveText = positiveNegativeNeutralText[positiveNegativeNeutralText["target"] == 4].sample(25000)

positiveText.loc[:, "target"] = "non-rascist/sexist"
positiveText = positiveText.loc[:, ["target", "text"]]

sexistText.loc[:, "offensive_language"] = "sexist"
racistText.loc[:, "Label"] = "racist"

racistText = racistText.loc[:, ["Label", "Text"]]
sexistText = sexistText.loc[:, ["offensive_language", "tweet"]]


racistText = racistText.rename(columns={"Label": "target", "Text": "text"})
sexistText = sexistText.rename(columns={"offensive_language": "target", "tweet": "text"})

fullDataset = pd.concat([sexistText, racistText, positiveText])

# CountVectorizerObject = CountVectorizer()
# X = CountVectorizerObject.fit_transform(fullDataset["text"])

# print(X)

def removeBeginningBadInformation(x):
    try:
        if "!" in x.split()[0]:
            try:
                return x[x.index(":")+2:]
            except:
                return x

        else:
            return x
    except:
        return x

def removeQuotations(x):
    try:
        if '"' in x.split()[0]:
            return x[2:-2]
        return x
    except:
        return x

fullDataset = fullDataset[~fullDataset["text"].str.contains("#porn")]


fullDataset['text'] = fullDataset['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
fullDataset['text'] = fullDataset['text'].apply(lambda x: removeBeginningBadInformation(x))
fullDataset['text'] = fullDataset['text'].apply(lambda x: ' '.join([word for word in x.split() if len(re.match("&*", word).group()) == 0]))
fullDataset['text'] = fullDataset['text'].apply(lambda x: removeQuotations(x))





cv = CountVectorizer()

fullDatasetTextList = fullDataset["text"].tolist()

X = fullDatasetTextList

# X = cv.fit_transform(fullDatasetTextList).toarray()
y = fullDataset.loc[:, "target"].tolist()


# import pickle
#
# cv2 = CountVectorizer()
#
# X_train, X_test, y_train, y_test = train_test_split(X, y)
#
#
# cv2.fit(X_train)
# X_train = cv2.transform(X_train)
#
# mNBModel = MultinomialNB()
#
# mNBModel.fit(X_train, y_train)
#
#
#
# pickle.dump(mNBModel, open("MultinomialNBModel2", 'wb'))
#
# pickle.dump(cv2, open("MultinomialNBCountVectorizer2", 'wb'))


# split train and test data

###################################33

cv = CountVectorizer()

X = cv.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y)

#
# mNBModel = MultinomialNB()
#
#
# mNBModel.fit(X_train, y_train)
# mNBModel.score(X_test, y_test)
#
#
# mNBModel.fit(X, y)
########################################


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout



# MAX NUMBER OF WORDS TO BE USED
MAX_NB_WORDS = 50000

# MAX NUMBER OF WORDS IN EACH MESSAGE
MAX_SEQUENCE_LENGTH = 250

EMBEDDING_DIM = 100

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(13, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 5
batch_size = 64

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
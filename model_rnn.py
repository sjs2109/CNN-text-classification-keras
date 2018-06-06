# 0. 사용할 패키지 불러오기
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from data_helpers import load_data
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
seed = 42
np.random.seed(seed)
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

max_features = 20000
text_max_words = 200

# 1. 데이터셋 생성하기

# 훈련셋과 시험셋 불러오기

x, y, vocabulary, _ = load_data()

dfRaw = pd.read_csv("./data/Womens Clothing E-Commerce Reviews_filtered.csv")
dfRec = dfRaw[['Review Text', 'Recommended IND']].dropna()
positive_examples = dfRec[dfRec['Recommended IND'] == 1]['Review Text'].tolist()
negative_examples = dfRec[dfRec['Recommended IND'] == 0]['Review Text'].tolist()

positive_labels = [1 for _ in positive_examples]
negative_labels = [0 for _ in negative_examples]
y = np.concatenate([positive_labels, negative_labels], 0)


# 데이터셋 전처리 : 문장 길이 맞추기
x = sequence.pad_sequences(x, maxlen=text_max_words)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(x, y):
    # 2. 모델 구성하기
    model = Sequential()
    model.add(Embedding(len(vocabulary), 1024))
    model.add(LSTM(1024))
    model.add(Dense(1, activation='sigmoid'))

    # 3. 모델 학습과정 설정하기
    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{acc:.4f}.hdf5', monitor='acc', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 4. 모델 학습시키기
    hist = model.fit(x[train], y[train], epochs=20, batch_size=32,callbacks=[checkpoint, early_stopping])

    scores = model.evaluate(x[test], y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
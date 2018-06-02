# 0. 사용할 패키지 불러오기
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from data_helpers import load_data
import pandas as pd
import numpy as np

max_features = 20000
text_max_words = 200

# 1. 데이터셋 생성하기

# 훈련셋과 시험셋 불러오기

x, _, vocabulary, _ = load_data()


dfRaw = pd.read_csv("./data/Womens Clothing E-Commerce Reviews_filtered.csv")
dfRec = dfRaw[['Review Text', 'Recommended IND']].dropna()
positive_examples = dfRec[dfRec['Recommended IND'] == 1]['Review Text'].tolist()
negative_examples = dfRec[dfRec['Recommended IND'] == 0]['Review Text'].tolist()

positive_labels = [1 for _ in positive_examples]
negative_labels = [0 for _ in negative_examples]
y = np.concatenate([positive_labels, negative_labels], 0)

x_temp, x_test, y_temp, y_test = train_test_split( x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split( x_temp, y_temp, test_size=0.2, random_state=42)

x_test = x_test.tolist()
y_test = y_test.tolist()

x_val = x_val.tolist()
y_val = y_val.tolist()


# 데이터셋 전처리 : 문장 길이 맞추기
x_train = sequence.pad_sequences(x_train, maxlen=text_max_words)
x_val = sequence.pad_sequences(x_val, maxlen=text_max_words)
x_test = sequence.pad_sequences(x_test, maxlen=text_max_words)

# 2. 모델 구성하기
model = Sequential()
model.add(Embedding(len(vocabulary), 128))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# 3. 모델 학습과정 설정하기
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val))

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('## evaluation loss and_metrics ##')
print(loss_and_metrics)
print('{0} : {1}'.format(model.metrics_names[1], loss_and_metrics[1]))
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from data_helpers import load_data
from sklearn.utils import shuffle
import numpy as np
from keras.callbacks import EarlyStopping

print('Loading data')
x, y, vocabulary, vocabulary_inv = load_data()

x, y = shuffle(x, y, random_state=42)
i = 0

x_split = [[], [], [], [], [], [], [], [], [], []]
y_split = [[], [], [], [], [], [], [], [], [], []]

for i in range(x.shape[0]):
    if i % 10 == 0:
        x_split[0].append(x[i])
        y_split[0].append(y[i])
    elif i % 10 == 1:
        x_split[1].append(x[i])
        y_split[1].append(y[i])
    elif i % 10 == 2:
        x_split[2].append(x[i])
        y_split[2].append(y[i])
    elif i % 10 == 3:
        x_split[3].append(x[i])
        y_split[3].append(y[i])
    elif i % 10 == 4:
        x_split[4].append(x[i])
        y_split[4].append(y[i])
    elif i % 10 == 5:
        x_split[5].append(x[i])
        y_split[5].append(y[i])
    elif i % 10 == 6:
        x_split[6].append(x[i])
        y_split[6].append(y[i])
    elif i % 10 == 7:
        x_split[7].append(x[i])
        y_split[7].append(y[i])
    elif i % 10 == 8:
        x_split[8].append(x[i])
        y_split[8].append(y[i])
    else:
        x_split[9].append(x[i])
        y_split[9].append(y[i])


result = []

sequence_length = x.shape[1] # 56
vocabulary_size = len(vocabulary_inv) # 18765
embedding_dim = 1024
filter_sizes = [3,4,5]
num_filters = 1024
drop = 0.5
epochs = 100
batch_size = 32

for i in range(10):
    x_temp = []
    y_temp = []
    x_test = []
    y_test = []
    x_test = np.asarray(x_split[i])
    y_test = np.asarray(y_split[i])

    for j in range(10):
        if j != i:
            x_temp = x_temp + x_split[j]
            y_temp = y_temp + y_split[j]

    x_temp = np.asarray(x_temp)
    y_temp = np.asarray(y_temp)
    x_train, x_val, y_train, y_val = train_test_split( x_temp, y_temp, test_size=0.2, random_state=42)

    # this returns a tensor
    print("Creating Model...")
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='he_uniform', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='he_uniform', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='he_uniform', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=2, activation='softmax')(dropout)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)

    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=4, verbose=2, mode='auto')
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    print("Traning Model...")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, callbacks=[checkpoint, early_stopping], validation_data=(x_val, y_val))  # starts training

    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
    print('## evaluation loss and_metrics ##')
    print(loss_and_metrics)
    print('{0} : {1}'.format(model.metrics_names[1], loss_and_metrics[1]))
    result.append(loss_and_metrics[1])


print('{0} : {1} {2} : {3}'.format("mean", np.mean(result) , "std" , np.std(result)))
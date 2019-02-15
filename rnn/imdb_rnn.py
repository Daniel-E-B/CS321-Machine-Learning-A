from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential, save_model, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, CuDNNLSTM, Dense, Flatten, Dropout, CuDNNGRU

(imdb_x_train, imdb_y_train), (imdb_x_test, imdb_y_test) = imdb.load_data()

cutoff = 500

imdb_x_train_padded = sequence.pad_sequences(imdb_x_train, maxlen=cutoff)
imdb_x_test_padded = sequence.pad_sequences(imdb_x_test, maxlen=cutoff)

imdb_index_offset = 3



imdb_lstm_model = Sequential()
imdb_lstm_model.add(Embedding(input_dim=len(imdb.get_word_index()) + imdb_index_offset,
                              output_dim=100,
                              input_length=cutoff))
# return_sequences tells the LSTM to output the full sequence, for use by the next LSTM layer. The final
# LSTM layer should return only the output sequence, for use in the Dense output layer
# gives the output for every timestep
imdb_lstm_model.add(CuDNNLSTM(units=5, return_sequences=True)) # 0
imdb_lstm_model.add(Dropout(0.5))
imdb_lstm_model.add(CuDNNLSTM(units=5, return_sequences=True)) # 1
imdb_lstm_model.add(Dropout(0.5))
imdb_lstm_model.add(CuDNNGRU(5))
imdb_lstm_model.add(Dropout(0.5))
imdb_lstm_model.add(Dense(units=1, activation='sigmoid')) # because at the end, we want one yes/no answer
imdb_lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

checkpt = ModelCheckpoint(filepath='imdb_lstm.hdf5', verbose=1, save_best_only=True)


# imdb_lstm_model.fit(imdb_x_train_padded, imdb_y_train, epochs=1000, batch_size=64, validation_data=(imdb_x_test_padded, imdb_y_test), callbacks=[checkpt])

imdb_lstm_model = load_model('imdb_lstm.hdf5')

imdb_lstm_scores = imdb_lstm_model.evaluate(imdb_x_test_padded, imdb_y_test)
print('loss: {} accuracy: {}'.format(*imdb_lstm_scores))
imdb_lstm_model.save('loss: {} accuracy: {}'.format(*imdb_lstm_scores) + '_rnn.h5')  # creates a HDF5 file 'my_model.h5'

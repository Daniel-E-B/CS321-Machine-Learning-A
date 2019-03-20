#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential, load_model
from keras.layers import Embedding, Conv1D, Dense, Flatten, Dropout, MaxPooling1D
from keras.datasets import reuters
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import model_to_dot
from gensim.models import word2vec
import numpy
from sklearn.preprocessing import LabelBinarizer
from IPython.display import SVG
(x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz", 
                                                         num_words=None, 
                                                         skip_top=0, 
                                                         maxlen=None, 
                                                         test_split=0.2, 
                                                         seed=113, 
                                                         start_char=1, 
                                                         oov_char=2, 
                                                         index_from=3)


# In[2]:


offset = 3
reuters_map = dict((index + offset, word) for (word, index) in reuters.get_word_index().items())
reuters_map[0] = 'PADDING'
reuters_map[1] = 'START'
reuters_map[2] = 'UNKNOWN'


# In[3]:


' '.join([reuters_map[word_index] for word_index in x_train[0]])


# In[4]:


train_sentences = [['PADDING'] + [reuters_map[word_index] for word_index in review] for review in x_train]
test_sentences = [['PADDING'] + [reuters_map[word_index] for word_index in review] for review in x_test]
# test_sentences


# In[5]:


reuters_wv_model = word2vec.Word2Vec(train_sentences + test_sentences + ['UNKNOWN'], min_count=1)


# In[6]:


reuters_wordvec = reuters_wv_model.wv
reuters_wv_model.wv.get_vector('snake')

reuters_wv_model.wv.vectors # list of word vectors


# In[ ]:





# In[7]:


# shorten and pad
lengths = [len(review) for review in x_train.tolist() + x_test.tolist()]
print('Longest review: {} Shortest review: {}'.format(max(lengths), min(lengths)))


# In[8]:


cutoff = 500
print('{} reviews out of {} are over {}.'.format(
    sum([1 for length in lengths if length > cutoff]), 
    len(lengths), 
    cutoff))


# In[9]:


from keras.preprocessing import sequence
x_train_padded = sequence.pad_sequences(x_train, maxlen=cutoff)
x_test_padded = sequence.pad_sequences(x_test, maxlen=cutoff)
len(x_train_padded)


# In[10]:


model = Sequential()
embedding_layer = reuters_wordvec.get_keras_embedding(train_embeddings=False)
embedding_layer.input_length = cutoff


# In[11]:


model.add(embedding_layer)


# In[12]:


# verify that embedding layer works the same as regular wordvec
model.predict(numpy.array([[reuters_wordvec.vocab["W"].index]]))[0][0] == reuters_wordvec["W"]


# In[13]:


y_train # 0 to 45 (46 categories)... need to onehot encode so that we can do softmax with a good fit


# In[14]:


onehot_enc = LabelBinarizer()
y_train_onehot = onehot_enc.fit_transform(y_train)
y_test_onehot = onehot_enc.fit_transform(y_test)


# In[15]:


model.add(Conv1D(filters=50, strides=2, kernel_size=5, activation='relu'))
model.add(Dropout(0.5))
model.add(Conv1D(filters=50, strides=2, kernel_size=7, activation='relu'))
model.add(MaxPooling1D(pool_size=3, strides=None, padding='valid', data_format='channels_last'))
model.add(Dropout(0.5))
model.add(Conv1D(filters=50, strides=2, kernel_size=5, activation='relu'))
model.add(Dropout(0.5))
model.add(Conv1D(filters=50, strides=2, kernel_size=7, activation='relu'))
model.add(MaxPooling1D(pool_size=3, strides=None, padding='valid', data_format='channels_last'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units=200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=46, activation='softmax')) # 46 topics
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])


# In[16]:


# for i in x_train_padded[0]:
#     print(reuters_wordvec.index2word[i]) # broke


# In[17]:


# fix the x_train_padded
for r_index, review in enumerate(x_train_padded):
    for i, word_index in enumerate(review):
        x_train_padded[r_index][i] = reuters_wordvec.vocab[reuters_map[word_index]].index

for r_index, review in enumerate(x_test_padded):
    for i, word_index in enumerate(review):
        x_test_padded[r_index][i] = reuters_wordvec.vocab[reuters_map[word_index]].index


# In[18]:


# for i in x_train_padded[0]:
#     print(reuters_wordvec.index2word[i]) # fixed


# In[19]:


checkpt = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
model.fit(x_train_padded, y_train_onehot, epochs=1000, batch_size=128, validation_data=(x_test_padded, y_test_onehot), callbacks=[checkpt])


# In[23]:


import matplotlib.pyplot as plt
plt.figure()
plt.hist(y_train, bins=46)
plt.show() # guess reuters has a lot of articles about the same topics


# In[22]:


# model = load_model("BEST_reuters_weights_64.hdf5") # trained 1000 epochs on bandersnatch
# SVG(model_to_dot(model).create(prog='dot', format='svg'))

scores = model.evaluate(x_test_padded, y_test_onehot)
print('loss: {} accuracy: {}'.format(*scores))


# In[ ]:





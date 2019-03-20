#!/usr/bin/env python
# coding: utf-8

# ## Installation
# 
# 1. If you haven't already installed Python3, get it from [Python.org](https://www.python.org/downloads/). **Tensorflow does not yet work with Python 3.7, so you _must_ get Python 3.6.** See https://github.com/tensorflow/tensorflow/issues/20517 for updates on 3.7 support.
# 1. If you haven't already installed Jupyter Notebook, run `python3 -m pip install jupyter`
# 1. In Terminal, cd to the folder in which you downloaded this file and run `jupyter notebook`. This should open up a page in your web browser that shows all of the files in the current directory, so that you can open this file. You will need to leave this Terminal window up and running and use a different one for the rest of the instructions.
# 1. Install the Gensim word2vec Python implementation: `pip3 install --upgrade gensim`
# 1. Get the trained model (1billion_word_vectors.zip) from me via airdrop or flashdrive and put it in the same folder as the ipynb file, the folder in which you are running the jupyter notebook command.
# 1. Unzip the trained model file. You should now have three files in the folder (if zip created a new folder, move these files out of that separate folder into the same folder as the ipynb file):
#     * 1billion_word_vectors
#     * 1billion_word_vectors.syn1neg.npy
#     * 1billion_word_vectors.wv.syn0.npy
# 1. If you didn't install keras last time, install it now
#     1. Install the tensorflow machine learning library by typing the following into Terminal:
#     `pip3 install --upgrade tensorflow`
#     1. Install the keras machine learning library by typing the following into Terminal:
#     `pip3 install keras`
# 

# ## Extra Details -- Do Not Do This
# This took awhile, which is why I'm giving you the trained file rather than having you do this. But just in case you're curious, here is how to create the trained model file.
# 1. Download the corpus of sentences from [http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz](http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz)
# 1. Unzip and unarchive the file: `tar zxf 1-billion-word-language-modeling-benchmark-r13output.tar.gz` 
# 1. Run the following Python code:
#     ```
#     from gensim.models import word2vec
#     import os
# 
#     corpus_dir = '1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled'
#     sentences = word2vec.PathLineSentences(corpus_dir)
#     model = word2vec.Word2Vec(sentences) # just use all of the default settings for now
#     model.save('1billion_word_vectors')
#     ```

# ## Documentation/Sources
# * [https://radimrehurek.com/gensim/models/word2vec.html](https://radimrehurek.com/gensim/models/word2vec.html) for more information about how to use gensim word2vec in general
# * [https://codekansas.github.io/blog/2016/gensim.html](https://codekansas.github.io/blog/2016/gensim.html) for information about using it to create embedding layers for neural networks.
# * [https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/](https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/) for information on sequence classification with keras
# * [https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html) for using pre-trained embeddings with keras (though the syntax they use for the model layers is different than most other tutorials I've seen).
# * [https://keras.io/](https://keras.io/) Keras API documentation

# ## Load the trained word vectors

# In[1]:


from gensim.models import word2vec
import keras.backend as K
from keras.callbacks import ModelCheckpoint


# Load the trained model file into memory

# In[2]:


wv_model = word2vec.Word2Vec.load('/root/projects/Daniel/cs321/1billion_word_vectors')


# Since we do not need to continue training the model, we can save memory by keeping the parts we need (the word vectors themselves) and getting rid of the rest of the model.

# In[3]:


wordvec = wv_model.wv
del wv_model


# ## Exploration of word vectors
# Now we can look at some of the relationships between different words.
# 
# Like [the gensim documentation](https://radimrehurek.com/gensim/models/word2vec.html), let's start with a famous example: king + woman - man

# In[4]:


wordvec.most_similar(positive=['king', 'woman'], negative=['man'])


# This next one does not work as well as I'd hoped, but it gets close. Maybe you can find a better example.

# In[5]:


wordvec.most_similar(positive=['panda', 'eucalyptus'], negative=['bamboo'])


# In[6]:


wordvec.most_similar(positive=['giraffe', 'short'], negative=[])


# Which one of these is not like the others?

# In[7]:


wordvec.doesnt_match(['red', 'purple', 'laptop', 'turquoise', 'ruby'])


# How far apart are different words?

# In[8]:


wordvec.distances('laptop', ['computer', 'phone', 'rabbit'])


# Let's see what one of these vectors actually looks like.

# In[9]:


wordvec['textbook']


# What other methods are available to us?

# In[10]:


help(wordvec)


# # Optional Exercise: Explore Word Vectors
# What other interesting relationship can you find, using the methods used in the examples above or anything you find in the help message?

# In[11]:


wordvec.most_similar(positive=['giraffe'], negative=['tall']) # girrafe thats not tall is a hippo


# In[12]:


wordvec.most_similar(positive=['giraffe', 'short']) # short giraffe is a zebra


# ## Using the word vectors in an embedding layer of a Keras model

# In[13]:


from keras.models import Sequential
import numpy


# You may have noticed in the help text for wordvec that it has a built-in method for converting into a Keras embedding layer.

# In[14]:


test_embedding_layer = wordvec.get_keras_embedding()
test_embedding_layer.input_length = 1


# In[15]:


embedding_model = Sequential()
embedding_model.add(test_embedding_layer)


# But how do we actually use this? If you look at the [Keras Embedding Layer documentation](https://keras.io/layers/embeddings/) you might notice that it takes numerical input, not strings. How do we know which number corresponds to a particular word? In addition to having a vector, each word has an index:

# In[16]:


wordvec.vocab['python'].index


# Let's see if we get the same vector from the embedding layer as we get from our word vector object.

# In[17]:


wordvec['python']


# In[18]:


embedding_model.predict(numpy.array([[30438]]))


# Looks good, right? But let's not waste our time when the computer could tell us definitively and quickly:

# In[19]:


embedding_model.predict(numpy.array([[wordvec.vocab['python'].index]]))[0][0] == wordvec['python']


# Now we have a way to turn words into word vectors with Keras layers. Yes! Time to get some data.

# ## The IMDB Dataset
# The [IMDB dataset](https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification) consists of movie reviews that have been marked as positive or negative. (There is also a built-in dataset of [Reuters newswires](https://keras.io/datasets/#reuters-newswire-topics-classification) that have been classified by topic.)

# In[20]:


from keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data()


# It looks like our labels consists of 0 or 1, which makes sense for positive and negative.

# In[21]:


print(y_train[0:9])
print(max(y_train))
print(min(y_train))


# But x is a bit more trouble. The words have already been converted to numbers -- numbers that have nothing to do with the word embeddings we spent time learning!

# In[22]:


x_train[0]


# Looking at the help page for imdb, it appears there is a way to get the word back. Phew.

# In[23]:


help(imdb)


# In[24]:


imdb_offset = 3
imdb_map = dict((index + imdb_offset, word) for (word, index) in imdb.get_word_index().items())
imdb_map[0] = 'PADDING'
imdb_map[1] = 'START'
imdb_map[2] = 'UNKNOWN'


# The knowledge about the initial indices came from [this stack overflow post](https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset) after I got gibberish when I tried to translate the first review, below. It looks coherent now!

# In[25]:


' '.join([imdb_map[word_index] for word_index in x_train[0]])


# ## Train our IMDB word vectors

# In[26]:


train_sentences = [['PADDING'] + [imdb_map[word_index] for word_index in review] for review in x_train]
test_sentences = [['PADDING'] + [imdb_map[word_index] for word_index in review] for review in x_test]


# In[27]:


imdb_wv_model = word2vec.Word2Vec(train_sentences + test_sentences + ['UNKNOWN'], min_count=1)


# In[28]:


imdb_wordvec = imdb_wv_model.wv
imdb_wv_model.wv.get_vector('snake')

imdb_wv_model.wv.vectors # list of word vectors


# ## Process the dataset
# For this exercise, we're going to keep all inputs the same length (we'll see how to do variable-length later). This means we need to choose a maximum length for the review, cutting off longer ones and adding padding to shorter ones. What should we make the length? Let's understand our data.

# In[29]:


lengths = [len(review) for review in x_train + x_test]
print('Longest review: {} Shortest review: {}'.format(max(lengths), min(lengths)))


# 2697 words! Wow. Well, let's see how many reviews would get cut off at a particular cutoff.

# In[30]:


cutoff = 500
print('{} reviews out of {} are over {}.'.format(
    sum([1 for length in lengths if length > cutoff]), 
    len(lengths), 
    cutoff))


# In[31]:


from keras.preprocessing import sequence
x_train_padded = sequence.pad_sequences(x_train, maxlen=cutoff)
x_test_padded = sequence.pad_sequences(x_test, maxlen=cutoff)


# ## Classification without using the pre-trained word vectors

# In[32]:


from keras.models import Sequential
from keras.layers import Embedding, Conv1D, Dense, Flatten
# x_train = list of words (review), y_train = positive or negative review


# Model definition.

# In[33]:


no_vector_model = Sequential()
no_vector_model.add(Embedding(input_dim=len(imdb_map), output_dim=100, input_length=cutoff)) # vectorizes the input
no_vector_model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
no_vector_model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
no_vector_model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
no_vector_model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
no_vector_model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
no_vector_model.add(Flatten())
no_vector_model.add(Dense(units=128, activation='relu'))
no_vector_model.add(Dense(units=1, activation='sigmoid')) # because at the end, we want one yes/no answer
no_vector_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])


# Train the model. __This takes awhile. You might not want to re-run it.__

# In[34]:


# no_vector_model.fit(x_train_padded, y_train, epochs=1, batch_size=64)


# Assess the model. __This takes awhile. You might not want to re-run it.__

# In[35]:


no_vector_scores = no_vector_model.evaluate(x_test_padded, y_test)
print('loss: {} accuracy: {}'.format(*no_vector_scores))


# # Exercise: Use the word vectors in a full model
# Using the knowledge about how the imdb dataset and the keras embedding layer represent words, as detailed above, define a model that uses the pre-trained word vectors from the imdb dataset rather than an embedding that keras learns as it goes along. You'll need to swap out the embedding layer and feed in different training data.
# 
# For any model that you try, take notes about the performance you see or anything you notice about the differences between each of them.
# 
# ## Other optional exercises:
# * Try using the 1billion vector word embeddings instead of the imdb vectors.
# * Try changing different hyperparameters of the model.
# * Try training with the reuters data.

# In[36]:


custom_embedding_layer = imdb_wordvec.get_keras_embedding(train_embeddings=False)
print(custom_embedding_layer.input_dim)
print(len(imdb_map))
print(custom_embedding_layer.input_length)
custom_embedding_layer.input_length = cutoff
custom_embedding_layer


# In[37]:


# for i in x_train_padded[0]:
#     print(imdb_wordvec.index2word[i]) # broke


# In[38]:


'''
 * fix mismatched indexes:
 * get the word vector for each index of the custom embedding layer.
 * find which index of the imdb wordvec has this wordvector
 * change the x's to use the correct indexes
'''
imdb_map[5] # make sense
imdb_wordvec.index2word[5] # weird (should be the same after this)
# for review in x_train:
#     for word_index in review:
# #         print(imdb_map[word_index])
#         print(imdb_wordvec.index2word[word_index])
#         if word_index > 1000:
#             break

    

# x train padded numbers > imdb map > word > get index of word in  > change the x train to that
# but imdb_map also has PADDING START and UNKNOWN
for r_index, review in enumerate(x_train_padded):
    for i, word_index in enumerate(review):
        x_train_padded[r_index][i] = imdb_wordvec.vocab[imdb_map[word_index]].index

for r_index, review in enumerate(x_test_padded):
    for i, word_index in enumerate(review):
        x_test_padded[r_index][i] = imdb_wordvec.vocab[imdb_map[word_index]].index


# In[39]:


# for i in x_train_padded[0]:
#     print(imdb_wordvec.index2word[i]) # fixed


# In[ ]:





# In[40]:


custom_embedding_model = Sequential()
custom_embedding_model.add(custom_embedding_layer)


# In[41]:


imdb_wordvec.vocab['python'].index
imdb_wordvec.index2word[88590]


# In[42]:


# making sure we get the same thing from wordvec as from layer:

custom_embedding_model.predict(numpy.array([[imdb_wordvec.vocab["W"].index]]))[0][0] == imdb_wordvec["W"]


# In[43]:


custom_embedding_model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
custom_embedding_model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
custom_embedding_model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
custom_embedding_model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
custom_embedding_model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
custom_embedding_model.add(Flatten())
custom_embedding_model.add(Dense(units=128, activation='relu'))
custom_embedding_model.add(Dense(units=1, activation='sigmoid'))
custom_embedding_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])


# In[44]:


custom_embedding_scores = custom_embedding_model.evaluate(x_test_padded, y_test)
print('loss: {} accuracy: {}'.format(*custom_embedding_scores))


# In[45]:

checkpt = ModelCheckpoint(filepath='custom_embedding_wordvectors_fixed.h5', verbose=1, save_best_only=False)

custom_embedding_model.fit(x_train_padded, y_train, epochs=15, batch_size=64, validation_data=(x_test_padded, y_test), callbacks=[checkpt])


# In[46]:


custom_embedding_scores = custom_embedding_model.evaluate(x_test_padded, y_test)
print('loss: {} accuracy: {}'.format(*custom_embedding_scores))


# In[47]:


from keras.models import load_model
custom_model = load_model('custom_embedding_wordvectors_fixed.h5') # trained for 20 epochs on bandersnatch... overfit very badly
# here is a line from training:
# fit line = custom_embedding_model.fit(x_train_padded, y_train, epochs=20, batch_size=64, validation_data=(x_test_padded, y_test))
# 25000/25000 [==============================] - 4s 163us/step - loss: 0.0205 - binary_accuracy: 0.9929 - val_loss: 1.2920 - val_binary_accuracy: 0.7893


# In[48]:


custom_embedding_scores = custom_model.evaluate(x_test_padded, y_test)
print('loss: {} accuracy: {}'.format(*custom_embedding_scores))


# In[49]:


custom_model.predict(x_test_padded[0:1])


# In[50]:


no_vector_model.predict(x_test_padded[0:1])


# In[51]:


y_test[0:1]


# In[52]:


custom_model.predict(x_test_padded[1:2])


# In[53]:


no_vector_model.predict(x_test_padded[1:2])


# In[54]:


y_test[1:2]


# overall seems to have learned (so it works)... although 78% from this model is not quite as good as the 85% from the no vector model (which only trained for one epoch...)

# In[55]:


# I trained my reuters model (with a different final layer) on imdb for 1000 epochs...
reuters_model = load_model("reuters-arch-imdb-data.hdf5")
reuters_model_scores = reuters_model.evaluate(x_test_padded, y_test)
print('loss: {} accuracy: {}'.format(*reuters_model_scores)) # its worse than custom_model


# In[ ]:





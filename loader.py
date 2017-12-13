import numpy as np
import matplotlib.pyplot as plt
from utils import Text20News # from: https://github.com/mdeff/cnn_graph/tree/master/lib

# 20news preprocessing
# Adapted from ChebNet code, MIT License, Copyright (c) 2016 Michael Defferrard
# We apply the exact same preprocessing.
def load_20news(data_home, top_words=1000, sparse=False,remove_short_documents=False, verbose=False):

  # Fetch dataset. Scikit-learn already performs some cleaning.
  remove = ('headers','footers','quotes')  # (), ('headers') or ('headers','footers','quotes')
  train = Text20News(data_home=data_home, subset='train', remove=remove)

  # Pre-processing: transform everything to a-z and whitespace.
  if verbose:
    print(train.show_document(1)[:400])
  train.clean_text(num='substitute')

  # Analyzing / tokenizing: transform documents to bags-of-words.
  #stop_words = set(sklearn.feature_extraction.text.ENGLISH_STOP_WORDS)
  # Or stop words from NLTK.
  # Add e.g. don, ve.
  train.vectorize(stop_words='english')
  if verbose:
    print(train.show_document(1)[:400])

  if remove_short_documents:
    # Remove short documents.
    if verbose:
      train.data_info(True)
    wc = train.remove_short_documents(nwords=20, vocab='full')
    if verbose:
      train.data_info()
      print('shortest: {}, longest: {} words'.format(wc.min(), wc.max()))
      plt.figure(figsize=(17,5))
      plt.semilogy(wc, '.');

    # Remove encoded images.
    def remove_encoded_images(dataset, freq=1e3):
        widx = train.vocab.index('ax')
        wc = train.data[:,widx].toarray().squeeze()
        idx = np.argwhere(wc < freq).squeeze()
        dataset.keep_documents(idx)
        return wc
    wc = remove_encoded_images(train)
    if verbose:
      train.data_info()
      plt.figure(figsize=(17,5))
      plt.semilogy(wc, '.');

  # Word embedding
  train.embed()

  # Feature selection.
  # Other options include: mutual information or document count.
  # freq = train.keep_top_words(1000, 20)
  # """Keep in the vocabulary the M words who appear most often."""
  M, Mprint = top_words, 20 
  freq = train.data.sum(axis=0)
  freq = np.squeeze(np.asarray(freq))
  idx = np.argsort(freq)[::-1]
  idx = idx[:M]
  train.keep_words(idx)
  if verbose:
    print('most frequent words')
    for i in range(Mprint):
        print('  {:3d}: {:10s} {:6d} counts'.format(i, train.vocab[i], freq[idx][i]))
  freq = freq[idx]
  if verbose:
    train.data_info()
    train.show_document(1)
    plt.figure(figsize=(17,5))
    plt.semilogy(freq);

  if remove_short_documents:
    # Remove documents whose signal would be the zero vector.
    wc = train.remove_short_documents(nwords=5, vocab='selected')
    if verbose:
      train.data_info(True)

  train.normalize(norm='l1')
  if verbose:
    train.show_document(1);

  # Test dataset.
  test = Text20News(data_home=data_home, subset='test', remove=remove)
  test.clean_text(num='substitute')
  test.vectorize(vocabulary=train.vocab)
  if verbose:
    test.data_info()
  if remove_short_documents:
    wc = test.remove_short_documents(nwords=5, vocab='selected')
  if verbose:
    print('shortest: {}, longest: {} words'.format(wc.min(), wc.max()))
    test.data_info(True)
  test.normalize(norm='l1')

  # Ready the data
  X_train = train.data.astype(np.float32)
  X_test = test.data.astype(np.float32)
  Y_train = train.labels
  Y_test = test.labels
  if sparse:
    X_train = csr_matrix((X_train.data, X_train.indices, X_train.indptr), shape=(X_train.shape[0], X_train.shape[1], 1))
    X_test = csr_matrix((X_test.data, X_test.indices, X_test.indptr), shape=(X_test.shape[0], X_test.shape[1], 1))
  else:
    X_train = np.array(X_train.todense()).reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = np.array(X_test.todense()).reshape((X_test.shape[0], X_test.shape[1], 1))
  input_shape = (X_train.shape[1], 1)

  # Convert class vectors to binary class matrices
  nb_classes = 20
  Y_train = np_utils.to_categorical(Y_train, nb_classes)
  Y_test = np_utils.to_categorical(Y_test, nb_classes)

  # Embeddings
  graph_data = train.embeddings.astype(np.float32)

  return (input_shape, nb_classes), (X_train, X_test, Y_train, Y_test), graph_data

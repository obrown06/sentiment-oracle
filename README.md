# sentiment-oracle

**sentiment-oracle** is a collection of sentiment classifiers and an API for serving their predictions in response to strings of English text. All classifiers were trained and evaluated on the [Amazon Reviews Dataset](https://www.kaggle.com/bittlingmayer/amazonreviews/data) and the [Yelp Reviews Dataset](https://www.yelp.com/dataset/download) and deployed after training on the [Stanford Treebank Rotten Tomatoes Dataset](https://nlp.stanford.edu/sentiment/).

The following classifiers are currently available to respond to API POST requests:

  * Multinomial Naive Bayes
  * Deep Neural Network, trained with SGD
  * Deep Neural Network, trained with Adam and built with PyTorch
  * LSTM Network, built with Keras/Tensorflow

Code for the following additional classifiers can be found in the source:

  * Bernoulli Naive Bayes
  * Logistic Regression, trained with SGD
  * LSTM Network, built with Pytorch

The api can be found [here](api.nlp-sentiment.com/predicts)!

## Data

## Preprocessing

## Feature Extraction

## Training

## Performance

## API

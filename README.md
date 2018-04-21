# sentiment-oracle

**sentiment-oracle** is a collection of sentiment classifiers and an API for serving their predictions in response to strings of English text. All classifiers were trained and evaluated on the [Amazon Reviews Dataset](https://www.kaggle.com/bittlingmayer/amazonreviews/data) and the [Yelp Reviews Dataset](https://www.yelp.com/dataset/download) and deployed after training on the [Stanford Treebank Rotten Tomatoes Dataset](https://nlp.stanford.edu/sentiment/).

The API currently supports requests to the following deployed classifiers:

  * Multinomial Naive Bayes
  * Feed Forward Neural Network, trained with SGD
  * Feed Forward Neural Network, trained with Adam and built with PyTorch
  * LSTM Network, built with Keras/Tensorflow

Additionally, code for the following classifiers can be found in the source:

  * Bernoulli Naive Bayes
  * Logistic Regression, trained with SGD
  * LSTM Network, built with Pytorch

The api can be accessed [here](http://api.nlp-sentiment.com/predicts), and a working demo [here](http://nicholasbrown.io/software)!

## Training, Testing, and Pickling

All training scripts can be found in the `train/` directory and invoked with `python3 train_script.py`. All scripts set parameters relevant for loading data and training their classifiers in their `data_info` and `classifier_info` dicts, respectively; these can be modified to your liking! In addition, each script uses the [pickle](https://docs.python.org/3/library/pickle.html) library to serialize its classifier and corresponding extractor at the location of the path specified in the source. This means you will overwrite your classifiers if you train multiple in a row without modifying the pickle path! Until I make some sort of CLI, you will have to modify the path (and any other desired parameters) in the source.

All testing scripts can be found in the `test/` directory and invoked with `python3 test_script.py`. Each script un-pickles a classifier and outputs its performance along with other relevant parameters specifying its architecture and training process.

## Data

We investigated three datasets:

1) The [Amazon Reviews](https://www.kaggle.com/bittlingmayer/amazonreviews/data) dataset, which consists of 4,000,000 [amazon.com](http://www.amazon.com) customer reviews, each labeled either negative or positive.
2) The [Yelp Reviews](https://www.yelp.com/dataset/download) dataset, which consists of 5,200,000 [Yelp](http://www.yelp.com) reviews, each labeled on a scale of 1 to 5.
3) The [Stanford Treebank Rotten Tomatoes](https://nlp.stanford.edu/sentiment/) dataset, which consists of 215,154 unique phrases parsed from 11,855 single sentences extracted from movie reviews on [Rotten Tomatoes](http://www.rottentomatoes.com), each labeled on a scale of 1 to 5.

The relative performance on these datasets is described below. The set of classifiers currently deployed were trained on the Treebank dataset.

## Preprocessing

Preprocessing steps on text strings include negation tracking, contraction expansion, and punctuation removal. We found that stopword removal diminished performance across all classifiers -- even after retaining those words with obvious connotations.

All preprocessing code lives in `/data/clean.py`.

**TODO**: tf-idf weighting

## Feature Extraction

Depending on the classifier, we use one of three kinds of features.

**words** (Bernoulli + Multinomial Naive Bayes)

Both Bernoulli and Multinomial Naive Bayes use individual words as features. MNB constructs conditional probabilities based on on the number of instances of a word in a class, while BNB does the same based on the number of documents in the class which contain the word. Both classifiers "train" by building a vocabulary of word counts across all documents in their corpus; feature extraction is as simple as splitting documents into individual tokens!

**Bag-Of-NGrams** (LR, NN)

Both the Logistic Regression and the two Feed-forward Neural Network classifiers use Bag-Of-NGrams as features.

In the Bag-Of-Words (BOW) model, we construct a vocabulary *V* from a subset of words in the document condition based on some defined criterion (in our case, frequency) and afterwards representing a document *D* as a vector of the number of occurrences in *D* of each of the words in *V*.

For example, under a vocabulary *V* : `{"it", "octupus", "best", "times"}`, we would represent a new document *D* : `"It was the best of times, it was the worst of times" ` with the vector `[2, 0, 1, 2]`.

The Bag-of-Ngrams model generalizes BOW to sequences of one **or more** adjacent words -- known as Ngrams. An example of a vocabulary containing both 1-grams and 2-grams would be *V* = `{"it", "octopus", "best", "it was", "was the"}`; under this vocabulary, the previous document *D* would be represented as `[2, 0, 1, 2, 2]`

Inevitably, lower order grams will be more common than their higher-order counterparts, so it is common when building a vocabulary to allocate percentages to tokens of different gram-numbers. We found that evenly splitting our vocabulary between 1-grams and 2-grams significantly boosted classifier performance; extending to higher order grams did not.

All extraction functionality is wrapped in the  extractor class (which can be pickled and reused) found in `/data/bag_of_ngrams_extractor.py`. NGram tokenization and sorting is performed with NLTK.


**Word Embeddings** (LSTM)

While the Bag-of-Ngrams model adeptly captures frequency of individual words, it fails to encode any information about the semantic relationships between words. Word Embeddings expresses this second category of information by converting each word to a high-dimensional vector of floating point numbers which, when trained based on some co-occurrence criteria over a sufficiently sized document corpus, represent some portion of its semantic meaning.

Word embeddings can be either trained along with the classifier in question or imported from some other corpus. I used pre-trained [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) to generate input for the PyTorch LSTM (see `/classifiers/lstm_pytorch.py` and `/data/glove_extractor.py`) but caution against following the evident procedure too closely, as its performance has been lackluster. On the other hand, the Keras LSTM (see `/classifiers/lstm_keras.py` and `/data/keras_extractor.py`) includes an Embedding layer initialized to random weights which is then trained in parallel with the LSTM layers. 

## Performance

## API

The API currently accepts GET and POST requests at the `/info` and `/predicts` endpoints. Visit [http://api.nlp-sentiment.com/predicts](http://api.nlp-sentiment.com/predicts) for instructions on POST request formatting.

The API was built with [Falcon](https://falconframework.org/), a (very) simple web framework geared towards the creation of REST APIs. It is currently deployed on a [Digital Ocean Droplet](https://www.digitalocean.com/products/droplets/) with [NGINX](https://www.nginx.com/) with [gunicorn](http://gunicorn.org/). I found [this tutorial](https://www.digitalocean.com/community/tutorials/how-to-deploy-falcon-web-applications-with-gunicorn-and-nginx-on-ubuntu-16-04) very helpful in getting everything set up.

Two notes:

1) All classifiers and extractors are un-pickled just once at app instantiation -- NOT upon each request -- which provides a significant speedup.
2) (relatedly) I removed the `--workers 3` flag in the Systemd ExecStart command, after trove of 504 errors after un-pickling classifiers at app instantiation.

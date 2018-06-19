# sentiment-oracle

**sentiment-oracle** is a collection of sentiment classifiers and an API for serving their predictions in response to strings of English text. All classifiers were trained and evaluated on the [Amazon Reviews Dataset](https://www.kaggle.com/bittlingmayer/amazonreviews/data) and the [Yelp Reviews Dataset](https://www.yelp.com/dataset/download) and deployed after training on the [Stanford Treebank Rotten Tomatoes Dataset](https://nlp.stanford.edu/sentiment/).

The API currently supports requests to the following deployed classifiers:

  * Multinomial Naive Bayes
  * Feed Forward Neural Network, trained with SGD
  * Feed Forward Neural Network, trained with Adam and built with PyTorch
  * LSTM Network, built with Keras/Tensorflow

Additional code for the following classifiers can also be found in the source:

  * Bernoulli Naive Bayes
  * Logistic Regression, trained with SGD
  * LSTM Network, built with Pytorch

The API can be accessed [here](http://api.nlp-sentiment.com/predicts); you can try out a working demo [here](http://nicholasbrown.io/demos/sentiment-oracle/)!

## Training, Testing, and Pickling

All training scripts can be found in the `train/` directory and invoked with `python3 train_script.py`. These scripts set parameters relevant for loading data and training their classifiers in their `data_info` and `classifier_info` dictionaries, respectively. Both of these can be modified to your liking!

Each script uses the [pickle](https://docs.python.org/3/library/pickle.html) library to serialize its classifier and corresponding extractor at the location of the path specified in the training script. This means you will overwrite data if you train multiple classifiers in a row without updating the pickle path! Until I make some sort of CLI, you will have to modify that path (along with any other desired parameters) in the source.

All testing scripts can be found in the `test/` directory and invoked with `python3 test_script.py`. Each script un-pickles a classifier and outputs 1) its performance and 2) other relevant parameters specifying its architecture and training process.

## Data

I investigated three datasets:

1) The [Amazon Reviews](https://www.kaggle.com/bittlingmayer/amazonreviews/data) dataset, which consists of 4,000,000 [amazon.com](http://www.amazon.com) customer reviews labeled either negative or positive.
2) The [Yelp Reviews](https://www.yelp.com/dataset/download) dataset, which consists of 5,200,000 [Yelp](http://www.yelp.com) reviews labeled on a scale of 1 to 5.
3) The [Stanford TreeBank Rotten Tomatoes](https://nlp.stanford.edu/sentiment/) dataset, which consists of 215,154 unique phrases parsed from 11,855 single sentences extracted from movie reviews on [Rotten Tomatoes](http://www.rottentomatoes.com) labeled on a scale of 1 to 5.

The relative performance on these datasets is described below. The set of classifiers currently deployed were trained on the TreeBank dataset.

## Preprocessing

Preprocessing operations performed on text strings include negation tracking, contraction expansion, and punctuation removal. I found that stopword removal diminished performance across all classifiers, even after retaining those stopwords with obvious connotations.

All preprocessing code lives in `/data/clean.py`.

**TODO**: tf-idf weighting

## Feature Extraction

Depending on the classifier, we train using one of three kinds of features.

**words** (Bernoulli + Multinomial Naive Bayes)

Both Bernoulli and Multinomial Naive Bayes use individual words as features. MNB constructs conditional probabilities based on on the number of times a word appears in a class, while BNB does the same based on the number of documents in the class which contain the word. Both classifiers "train" by updating a vocabulary of word counts across all documents in their corpus. For such a simple model, feature extraction is as simple as splitting documents into individual tokens!

**Bag-Of-Ngrams** (LR, NN)

Both the Logistic Regression and the two Feed-forward Neural Network classifiers use Bag-Of-Ngrams as features.

In the Bag-Of-Words (BOW) model, we construct a vocabulary *V* from a subset of words in the document condition based on some defined criterion (in our case, frequency) and afterwards representing a document *D* as a vector of the number of occurrences in *D* of each of the words in *V*.

For example, under a vocabulary *V* : `{"it", "octupus", "best", "times"}`, we would represent a new document *D* : `"It was the best of times, it was the worst of times" ` with the vector `[2, 0, 1, 2]`.

The Bag-of-Ngrams model generalizes BOW to sequences of one **or more** adjacent words -- known as Ngrams. An example of a vocabulary containing both 1-grams and 2-grams would be *V* = `{"it", "octopus", "best", "it was", "was the"}`; under this vocabulary, the previous document *D* would be represented as `[2, 0, 1, 2, 2]`

Inevitably, lower order grams will be more common than their higher-order counterparts, so it is common to reserve percentages within a vocabulary for tokens of particular gram-numbers. I found that evenly splitting the vocabulary between 1-grams and 2-grams significantly boosted classifier performance; extending to higher order grams did not.

All extraction functionality is wrapped in the  extractor class (which can be pickled and reused) found in `/data/bag_of_ngrams_extractor.py`. Ngram tokenization and sorting is performed with NLTK.


**Word Embeddings** (LSTM)

While the Bag-of-Ngrams model adeptly captures frequency of individual words, it fails to encode any information about the semantic relationships between them. Word Embeddings capture this second category of information by converting each word to a high-dimensional vector of floating point numbers which, when trained based on some co-occurrence criteria over a sufficiently sized document corpus, represent some portion of its semantic meaning.

Word embeddings can be either trained along with the classifier in question or imported from some other corpus. I used pre-trained [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) to generate input for the PyTorch LSTM (see `/classifiers/lstm_pytorch.py` and `/data/glove_extractor.py`) (a note, however: the classifier's performance has been lackluster, so I caution against following my procedure too closely). On the other hand, the Keras LSTM (see `/classifiers/lstm_keras.py` and `/data/keras_extractor.py`) includes an Embedding layer initialized to random weights which is then trained in parallel with the LSTM layers.

## Performance

Classifiers were trained and validated on three datasets: the [Amazon Reviews Dataset](https://www.kaggle.com/bittlingmayer/amazonreviews/data), the [Yelp Reviews Dataset](https://www.yelp.com/dataset/download), and the [Stanford Treebank Rotten Tomatoes Dataset](https://nlp.stanford.edu/sentiment/). As both of the 1-to-5 datasets are significantly
unbalanced (the Yelp dataset towards ratings of '4' and '5' and the Rotten Tomatoes dataset towards ratings of '3'), I have also evaluated their performance on artificially balanced subsets of both. In addition, I have included the binary classification performance of all classifiers on both of the '1' to '5' datasets; these were generated by excluding neutral '3' ratings and consolidating classes '1' and '2' and classes '4' and '5'. All listed percentages represent accuracy values over the test set.

**Binary**

| Classifier          | Amazon | Yelp  | RT    |
| ----------          | ------ | ------| ------|
| Multinomial NB      | 84.7%  | 87.9% | 81.6% |
| Logistic Regression | 85.7%  | 88.2% | 75.2% |
| Feed Forward (SGD)  | 86.6%  | 89.4% | 76.0% |       
| Feed Forward (Adam) | 88.2%  | 89.4% | 75.1% |
| LSTM                | 90.3%  | 93.0% | 83.5% |

**Fine-Grained**

| Classifier          | Yelp  | RT    |
| ----------          | ------| ------|
| Multinomial NB      | 60.5% | 61.4% |
| Logistic Regression | 62.7% | 58.8% |
| Feed Forward (SGD)  | 62.9% | 58.2% |
| Feed Forward (Adam) | 63.1% | 59.1% |
| LSTM                | 66.3% | 64.9% |

**Fine-Grained + Balanced**

| Classifier          | Yelp  | RT    |
| ----------          | ------| ------|
| Multinomial NB      | 54.7% | 41.9% |
| Logistic Regression | 55.3% | 37.8% |
| Feed Forward (SGD)  | 52.5% | 37.1% |
| Feed Forward (Adam) | 53.2% | 40.1% |
| LSTM                | 59.4% | 47.7% |


Some notes:

* Best performance across all datasets was achieved by the LSTM classifier, whose accuracy placed within the top 5% of submissions for the [Sentiment Analysis on Movie Reviews Kaggle Competition](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/leaderboard) (though still well shy of results achieved with recursive networks trained on deep semantic structures; see [Socher et al](https://nlp.stanford.edu/sentiment/)).
* MNB achieves best performance on the RT dataset. A possible reason for this is the relative brevity of documents in that dataset.
* All classifiers perform worse on the balanced RT and Yelp datasets, but the effect is most pronounced for deep networks.

## API

The API currently accepts GET and POST requests at the `/info` and `/predicts` endpoints. Visit [http://api.nlp-sentiment.com/predicts](http://api.nlp-sentiment.com/predicts) for instructions on POST request formatting.

The API was built with [Falcon](https://falconframework.org/), a simple web framework geared towards the creation of REST APIs. It is currently deployed on a [Digital Ocean Droplet](https://www.digitalocean.com/products/droplets/) with [NGINX](https://www.nginx.com/) and [gunicorn](http://gunicorn.org/). I found [this tutorial](https://www.digitalocean.com/community/tutorials/how-to-deploy-falcon-web-applications-with-gunicorn-and-nginx-on-ubuntu-16-04) very helpful in getting everything set up.

Two notes:

1) All classifiers and extractors are un-pickled just once at app instantiation, and NOT upon each request. This provides a significant performance boost.
2) (relatedly) I removed the `--workers 3` flag in the Systemd ExecStart command after encountering a trove of 504 errors upon un-pickling classifiers at app instantiation.

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

Preprocessing steps on text strings include negation tracking, contraction expansion, and punctuation removal. We found that stopword removal resulted in diminished performance across all classifiers even after retaining obviously those with obvious connotations.

All preprocessing code lives in `/data/clean.py`.

**TODO**: tf-idf weighting

## Feature Extraction

Depending on the classifier, we used one of three kinds of features.

**words**

The Bernoulli and Multinomial

**Bag of Words**

**Word Embeddings**


TODO: tf-idf weighting

## Performance

## API

The API currently accepts GET and POST requests at the `/info` and `/predicts` endpoints. Visit [http://api.nlp-sentiment.com/predicts](http://api.nlp-sentiment.com/predicts) for instructions on POST request formatting.

The API was built with [Falcon](https://falconframework.org/), a (very) simple web framework geared towards the creation of REST APIs. It is currently deployed on a [Digital Ocean Droplet](https://www.digitalocean.com/products/droplets/) with [NGINX](https://www.nginx.com/) with [gunicorn](http://gunicorn.org/). I found [this tutorial](https://www.digitalocean.com/community/tutorials/how-to-deploy-falcon-web-applications-with-gunicorn-and-nginx-on-ubuntu-16-04) very helpful in getting everything set up.

Two notes:

1) All classifiers and extractors are un-pickled just once at app instantiation -- NOT upon each request -- which provides a significant speedup.
2) (relatedly) I removed the `--workers 3` flag in the Systemd ExecStart command, after trove of 504 errors after un-pickling classifiers at app instantiation.

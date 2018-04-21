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

## Setup

## Training, Testing, and Pickling

All training scripts can be found in the `train/` directory and invoked with `python3 NAME_OF_SCRIPT.py`. Each scrip

## Data

All classifiers were trained with

## Preprocessing

Preprocessing

## Feature Extraction



## Performance

## API

The API currently accepts GET and POST requests at the `/info` and `predicts` endpoints; visit [http://api.nlp-sentiment.com/predicts](http://api.nlp-sentiment.com/predicts) for instructions on POST request formatting.

The API was built with [Falcon](https://falconframework.org/), a (very) simple web framework geared towards the creation of REST APIs. It is currently deployed on a [Digital Ocean Droplet](https://www.digitalocean.com/products/droplets/) with [NGINX](https://www.nginx.com/) with [gunicorn](http://gunicorn.org/). I found [this tutorial](https://www.digitalocean.com/community/tutorials/how-to-deploy-falcon-web-applications-with-gunicorn-and-nginx-on-ubuntu-16-04) very helpful in getting everything set up.

Two notes:

1) All classifiers and extractors are un-pickled just once at app instantiation -- NOT upon each request -- which provides a significant speedup.
2) (relatedly) I removed the `--workers 3` flag in the Systemd ExecStart command, after trove of 504 errors after un-pickling classifiers at app instantiation.

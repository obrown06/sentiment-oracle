train_texts, train_labels = strip_labels(train_reviews[0:int(len(train_reviews) / 1000)])
test_texts, test_labels = strip_labels(test_reviews[0:int(len(test_reviews) / 1000)])

lr_classifier = LogisticRegressionSAClassifier(pre_process(train_texts), train_labels)

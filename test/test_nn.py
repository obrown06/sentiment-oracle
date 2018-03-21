train_texts, train_labels = strip_labels(test_reviews[0:int(len(train_reviews) // 50)])
test_texts, test_labels = strip_labels(train_reviews[0:int(len(test_reviews) / 50)])

pre_processed_train_texts = pre_process(train_texts)
pre_processed_test_texts = pre_process(test_texts)

feature_set = build_feature_set(pre_process(train_texts))

input_matrix_train = input_matrix(pre_processed_train_texts, feature_set)
input_matrix_test = input_matrix(pre_processed_test_texts, feature_set)

dn = DeepNet()
layer_dims = [2000, 20, 5, 1]
dn.train(input_matrix_train, np.asarray(train_labels), layer_dims, 0.5, 3000, "batch")

accuracy, precision, recall, specificity = dn.test(input_matrix_test, np.asarray(test_labels), 0.5)

print("accuracy", accuracy)
print("precision", precision)
print("recall", recall)
print("specificity", specificity)

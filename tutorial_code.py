import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import time


###--Single Layer Class
class Perceptron(tf.Module):
    def __init__(self, weights, biases, name=None):
        super().__init__(name=name)
        self.weights = tf.Variable(weights, name="weights")
        self.biases = tf.Variable(biases, name='biases')
    def __call__(self, x):
        return tf.matmul(x, self.weights) + self.biases

###--Multi-Layer Perceptron Class
class MLP(tf.Module):
    def __init__(self, network, name=None):
        super().__init__(name=name)
        self.layers = self.build_model(network)

    ###--NEW--Calling Method
    @tf.function
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = tf.sigmoid(x)
        return x

    ###--NEW--Generalized Layer Creation
    def build_model(self, network):
        layers = []
        for i in range(len(network) - 1):
            (w, b) = self.build_dense_layer(network[i], network[i + 1])
            layers.append(Perceptron(w, b))
        return layers

    def build_dense_layer(self, nodes_in, nodes_out):
        weights = tf.random.normal([nodes_in,nodes_out], 0,1, dtype="float32")
        biases = tf.random.normal([nodes_out], 0,1, dtype="float32")
        return (weights, biases)

(train, lbtrain), (test, lbtest) = keras.datasets.mnist.load_data()
num_classes = 10

###--NEW--Collected all data used only for arguments to
        #def test_label_count()
test_labels = lbtest
all_labels = tf.concat([lbtrain, lbtest], 0)

train = train.reshape(60000, 784).astype("float32") / 255
test = test.reshape(10000, 784).astype("float32") / 255

lbtrain = tf.keras.utils.to_categorical(lbtrain, num_classes=num_classes, dtype="float32")
lbtest = tf.keras.utils.to_categorical(lbtest, num_classes=num_classes, dtype="float32")

###--NEW--Collects all MNIST samples and labels for K-Fold Cross-Validation
mnist = tf.concat([train, test], 0)
lbmnist = tf.concat([lbtrain, lbtest], 0)

@tf.function
def loss(predicted_y, target_y):
  return tf.reduce_mean(tf.square(predicted_y - target_y))

@tf.function
def train(model, x, y, epoch, learning_rate):
    with tf.GradientTape() as tape:
        current_loss = loss(model(x), y)
        ###--NEW--Collects weights and biases for tf.gradient()
        collect_var = []
        for layer in model.layers:
            collect_var.append(layer.weights)
            collect_var.append(layer.biases)
        # Use GradientTape to calculate the gradients with respect to weights and biases
        grad = tape.gradient(current_loss, collect_var)
        ###--NEW--Updates weights and Biases Layer by Layer
        c = 0
        for i in range(0, len(collect_var), 2):
            model.layers[c].weights.assign_sub(learning_rate * grad[i])
            model.layers[c].biases.assign_sub(learning_rate * grad[i + 1])
            c += 1

def training_loop(model, train_dataset, learning_rate, epochs):
    l = loss(model(test), lbtest)
    results = [l]
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))
        # Iterate over the batches of the dataset.
        for batch, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            train(model, x_batch_train, y_batch_train, epoch, learning_rate)
        
        l = loss(model(test), lbtest)
        results.append(l)
    return results

@tf.function
def testing_loop(model, test, lbtest):
    a = tf.keras.metrics.categorical_accuracy(model(test), lbtest)
    correct = float(tf.math.count_nonzero(a))
    total = float(len(test))
    accuracy = (correct/total) * 100.0
    return accuracy

###--NEW--Counts number of predicted labels
def test_label_count(model, test, lbtest, labels):
    d = {i:0 for i in range(num_classes)}
    a = tf.keras.metrics.categorical_accuracy(model(test), lbtest).numpy()
    for i in range(len(a)):
        if a[i] == 1:
            d[int(labels[i])] += 1
    correct = float(tf.math.count_nonzero(a))
    total = float(len(test))
    accuracy = (correct/total) * 100.0
    return (accuracy, d)

###--NEW--Evaluates model using correct method based on the value for k
def evaluate(model, dataset, labels, epochs=10, learning_rate=0.9, batch_size=10, k=5):
    num_samples = dataset.shape[0]
    errors = []
    if k > 1 and k < num_samples:
        errors = cross_validation(model, dataset, labels, epochs, learning_rate, batch_size, k)
    else:
        errors = single_split(model, dataset, labels, epochs, learning_rate, batch_size)
    return errors

###--NEW--Evaluates without Cross-Validation
def single_split(model, dataset, labels, epochs, learning_rate, batch_size):
    ###SPLIT DATASET INTO 7 SECTIONS
    dataset = tf.split(dataset, 7, 0)
    labels = tf.split(labels, 7, 0)
    ##CREATE 60,000 TRAINING IMAGES AND 10,000 TESTING IMAGES
    (train, lbtrain), (test, lbtest) = split_data(dataset, labels, 6)
    train_dataset = tf.data.Dataset.from_tensor_slices((train, lbtrain))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    #TEST THE MODEL
    startTime = time.time()
    losses = training_loop(model, train_dataset, learning_rate, epochs)
    accuracy = testing_loop(model, test, lbtest)
    executionTime = (time.time() - startTime)
    #CALCULATE THE RESULTS
    losses = np.array(losses)
    avg_all_errors = np.average(losses, 0)
    print("ACCURACY: %0.2f" % (accuracy))
    print("ERROR: %0.6f" % (losses[-1]))
    print("TOTAL RUN TIME (sec): %0.2fs" % (float(executionTime)))
    return losses

###--NEW--Evaluates without Cross-Validation
def cross_validation(model, dataset, labels, epochs, learning_rate, batch_size, k):
    ##NUMBER OF IMAGES
    len_dataset = dataset.shape[0]
    ##SPLIT THE DATA INTO AN EVEN NUMBER OF SECTIONS
    k_data = tf.split(dataset, k, 0)
    k_labels = tf.split(labels, k, 0)
    ##FOR COLLECTING RESULTS
    accuracies = []
    errors = []
    all_errors = []
    run_times = []
    ##ITERATE THE CROSS VALIDATION
    for i in range(0, k):
        print("K-Fold %2d" % (i + 1))
        ##CREATE TRAINING AND TESTING DATA
        (train, lbtrain), (test, lbtest) = split_data(k_data, k_labels, i)
        train_dataset = tf.data.Dataset.from_tensor_slices((train, lbtrain))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        #TEST THE MODEL
        startTime = time.time()
        losses = training_loop(model, train_dataset, learning_rate, epochs)
        accuracy = testing_loop(model, test, lbtest)
        executionTime = (time.time() - startTime)
        #TABULATE RESULTS
        all_errors.append(losses)
        errors.append(losses[-1].numpy())
        accuracies.append(accuracy)
        run_times.append(executionTime)
    #CALCULATE THE RESULTS
    avg_acc = float(sum(accuracies)) / float(len(accuracies))
    avg_error = float(sum(errors)) / float(len(errors))
    avg_runtime = float(sum(run_times)) / float(len(run_times))
    all_errors = np.array(all_errors)
    avg_all_errors = np.average(all_errors, 0)
    variance = np.var(np.array(accuracies))
    print("ACCURACY: %0.2f" % (avg_acc))
    print("ERROR: %0.6f" % (avg_error))
    print("VARIANCE: %0.2f" % (variance))
    print("ACCURACIES: ", np.array(accuracies))
    print("AVERAGE RUN TIME PER FOLD: %0.2fs, TOTAL RUN TIME (sec): %0.2fs" % (avg_runtime, float(sum(run_times))))
    return avg_all_errors

###--NEW--Returns the desired training and testing split    
        ###k_data and k_labels must be split into k-sections
def split_data(k_data, k_labels, split):
    assert(split <= len(k_labels))
    test = k_data[split]
    lbtest = k_labels[split]
    train = tf.concat(k_data[0:split] + k_data[split + 1: len(k_data)], 0)
    lbtrain = tf.concat(k_labels[0:split] + k_labels[split + 1: len(k_labels)], 0)
    return (train, lbtrain), (test, lbtest)

###--NEW--Generates Graph from Figure 21
def plot_errors():
    lrs = [0.9, 0.1, 0.01]
    colors = ['blue', 'green', 'orange']
    epochs = 25
    X = [i for i in range(epochs+1)]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i in range(len(lrs)):
        mlp = MLP([784, 15, 10])
        results = evaluate(mlp, mnist, lbmnist, epochs=epochs, learning_rate=lrs[i], batch_size=10, k=5)

        ax.plot(X, results, color=colors[i], label='lr='+str(lrs[i]))

    ax.legend()
    plt.title('Learning Rate Comparison (' + str(epochs) +  ' epochs)')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.show()

#HOW TO CALL METHOD FOR CROSS-VALIDATION

mlp = MLP([784, 10, 100, 400, 200, 50, 10])
evaluate(mlp, mnist, lbmnist, epochs=25, learning_rate=0.9, batch_size=10, k=5)

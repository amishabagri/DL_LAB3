# Convert the provided TF1-style CNN code to TensorFlow 2.x compatible version

import tensorflow as tf
import numpy as np
import time

seed = 1234
tf.random.set_seed(seed)
np.random.seed(seed)

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train / 255.0, axis=-1).astype(np.float32)
x_test = np.expand_dims(x_test / 255.0, axis=-1).astype(np.float32)
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

batch_size = 64
hidden_size = 100
learning_rate = 0.01
output_size = 10

class CNN(tf.keras.Model):
    def __init__(self, hidden_size, output_size):
        super(CNN, self).__init__()
        self.conv = tf.keras.layers.Conv2D(30, (5, 5), padding='same', activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_size)

    def call(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

def compute_accuracy(logits, labels):
    preds = tf.argmax(logits, axis=1)
    true = tf.argmax(labels, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(preds, true), tf.float32))

model = CNN(hidden_size, output_size)
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

print("\\n🧠 Starting Training...\\n")
total_start = time.time()

for epoch in range(4):
    epoch_start = time.time()
    avg_loss = 0.0
    steps = 0
    for images, labels in train_dataset:
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss = loss_fn(labels, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        avg_loss += loss.numpy()
        steps += 1

    train_logits = model(x_train, training=False)
    train_acc = compute_accuracy(train_logits, y_train) * 100
    epoch_time = time.time() - epoch_start
    print(f"🕒 Epoch {epoch+1} - Time: {epoch_time:.2f}s - Loss: {avg_loss/steps:.4f} - Accuracy: {train_acc:.2f}%")

total_time = time.time() - total_start
print(f"🎯 Total Training Time: {total_time:.2f} seconds")
print(f"⏱️ Avg Time per Epoch: {total_time/4:.2f} seconds")

test_logits = model(x_test, training=False)
test_acc = compute_accuracy(test_logits, y_test) * 100
print(f"🔍 Test Accuracy: {test_acc:.2f}%")


import tensorflow as tf
import numpy as np

# Define the parameters of the system
n = 10  # number of variables
k = np.random.rand(n)  # decay rates
W = np.random.rand(n, n)  # interaction matrix

print(W) #randomly generated initial interaction matrix

print(k) #randomly generated initial decay rate

# Define the neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(n, activation='sigmoid'),
    tf.keras.layers.Dense(n)
])

# Define the loss function
def loss_fn(y_true, y_pred, W, k):
    dydt_pred = tf.reduce_sum(W * y_pred, axis=1) - k * y_pred
    return tf.reduce_mean(tf.square(dydt_pred - y_true))

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Train the model
for i in range(1000):
    y = np.random.rand(n)  # initial condition
    with tf.GradientTape(persistent=True) as tape:
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        dydt_pred = model(tf.expand_dims(y, axis=-1))
        loss = loss_fn(dydt_pred, tf.zeros(n), W, k)
    dL_dW, dL_dk = tape.gradient(loss, [model.trainable_variables[2], model.trainable_variables[3]])
    W -= 0.001 * dL_dW #Gradient descent
    k -= 0.001 * dL_dk #Gradient descent
    del tape  # Release resources used by the tape

from keras.utils import plot_model

print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

print(W) #updated interaction matrix after training the model

print(k) #updated decay rate after training the model

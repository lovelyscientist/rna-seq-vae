from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

class VAE(tf.keras.Model):
  def __init__(self, latent_dim, input_shape):
    super(VAE, self).__init__()
    self.latent_dim = latent_dim

    self.inference_net = tf.keras.Sequential(
      [
          tf.keras.layers.InputLayer(input_shape=input_shape),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dense(self.latent_dim + self.latent_dim, kernel_initializer='glorot_uniform', activation=tf.nn.relu)
      ]
    )

    print(input_shape)

    self.generative_net = tf.keras.Sequential(
        [
          tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
          tf.keras.layers.Dense(input_shape[0],  kernel_initializer='glorot_uniform', activation=tf.nn.sigmoid)
        ]
    )


  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps)

  def encode(self, x):
    mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.generative_net(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs

    return logits


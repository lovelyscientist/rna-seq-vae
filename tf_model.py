from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

class VAE(tf.keras.Model):
  def __init__(self, latent_dim, input_shape):
    super(VAE, self).__init__()
    self.latent_dim = latent_dim


    input_features =  tf.keras.layers.InputLayer(input_shape=input_shape)
    input_labels = tf.keras.layers.InputLayer(input_shape=(1,))

    print(type([input_features, input_labels]))
    print(type(input_labels))

    concatenated_inputs = tf.keras.layers.concatenate([input_features,input_labels], axis=1)
    #concatenated_inputs = tf.keras.layers.Concatenate()([input_features, input_labels])
    batch_norm = tf.keras.layers.BatchNormalization()(concatenated_inputs)
    dense = tf.keras.layers.Dense(self.latent_dim + self.latent_dim,
                                  kernel_initializer='glorot_uniform', activation=tf.nn.relu)(batch_norm)

    inference_functional = tf.keras.models.Model(inputs=[input_features, input_labels], outputs=dense)
    self.inference_net = tf.keras.Sequential(layers=inference_functional.layers)

    print(input_shape)

    latent_input = tf.keras.layers.Input(shape=(self.latent_dim,))
    concatenated_inputs_gen = tf.keras.layers.concatenate([latent_input, input_labels], axis=1)
    #concatenated_inputs_gen = tf.keras.layers.Concatenate()([latent_input, input_labels])
    output =  tf.keras.layers.Dense(input_shape[0],
                                    kernel_initializer='glorot_uniform', activation=tf.nn.sigmoid)(concatenated_inputs_gen)

    generative_functional = tf.keras.models.Model(inputs=[latent_input, input_labels], outputs=output)
    self.generative_net = tf.keras.Sequential(layers=generative_functional.layers)


  @tf.function
  def sample(self, y, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, y)

  def encode(self, x, y):
    mean, logvar = tf.split(self.inference_net([x, y]), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, y, apply_sigmoid=False):
    logits = self.generative_net(z, y)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs

    return logits


import tensorflow as tf
from VariationalAutoencoder import VAE
import numpy as np
import os
import time
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
from gtex_loader import get_gtex_dataset
from pprint import pprint

writer = tf.summary.create_file_writer("./logs/run3")

(train_samples, _), (test_samples, _) = get_gtex_dataset()

train_samples = train_samples.astype('float32')
test_samples = test_samples.astype('float32')

# Normalizing the images to the range of [0., 1.]
#train_samples /= 255.
#test_samples /= 255.

# Binarization
#train_samples[train_samples >= .5] = 1.
#train_samples[train_samples < .5] = 0.
#test_samples[test_samples >= .5] = 1.
#test_samples[test_samples < .5] = 0.

TRAIN_BUF = len(train_samples)
BATCH_SIZE = 50
TEST_BUF = len(test_samples)
epochs = 100
latent_dim = 16
num_examples_to_generate = 6

train_dataset = tf.data.Dataset.from_tensor_slices(train_samples).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_samples).shuffle(TEST_BUF).batch(BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam(1e-5)

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

@tf.function
def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)

  '''cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent) # axis=[1, 2, 3]
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)'''

  reconstruction_loss = -tf.reduce_sum(x * tf.math.log(1e-10 + x_logit)+ (1 - x) * tf.math.log(1e-10 + 1 - x_logit))
  latent_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(mean), 1)

  return tf.reduce_mean(reconstruction_loss + latent_loss)

@tf.function
def compute_apply_gradients(model, x, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])

model = VAE(latent_dim, train_samples[0].shape)

def generate_and_save_images(model, epoch, test_input):
  predictions = model.sample(test_input)
  fig = plt.figure(figsize=(4,4))

  # tight_layout minimizes the overlap between 2 sub-plots
  #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  #plt.show()

#generate_and_save_images(model, 0, random_vector_for_generation)

def training():
  for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
      compute_apply_gradients(model, train_x, optimizer)
    end_time = time.time()
    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
      loss(compute_loss(model, test_x))
    elbo = -loss.result()
    with writer.as_default():
      tf.summary.scalar('elbo', data=elbo, step=epoch)
      tf.summary.histogram('inference first dense layer', data=model.inference_net.layers[0].trainable_weights[0], step=epoch)
      tf.summary.histogram('inference second dense layer', data=model.inference_net.layers[1].trainable_weights[0],
                           step=epoch)
      tf.summary.histogram('inference output layer', data=model.inference_net.layers[2].trainable_weights[0],
                           step=epoch)
      tf.summary.histogram('generative first dense layer', data=model.generative_net.layers[0].trainable_weights[0],
                           step=epoch)
      tf.summary.histogram('generative second dense layer', data=model.generative_net.layers[1].trainable_weights[0],
                           step=epoch)
      tf.summary.histogram('generative output layer', data=model.generative_net.layers[2].trainable_weights[0],
                           step=epoch)
    #display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, '
          'time elapse for current epoch {}'.format(epoch,
                                                    elbo,
                                                    end_time - start_time))
    #print(epoch)
    if epoch == epochs:
        random_vector_for_generation = tf.random.normal(
            shape=[num_examples_to_generate, latent_dim])
        predictions = model.sample(random_vector_for_generation)
        for p in predictions:
            print(p)

training()

def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

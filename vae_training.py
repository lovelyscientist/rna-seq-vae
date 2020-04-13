import tensorflow as tf
from model import VAE
import time
from gtex_loader import get_gtex_dataset
import numpy as np
from pprint import pprint

writer = tf.summary.create_file_writer("./logs/run3")

(train_samples, _), (test_samples, _), df_values = get_gtex_dataset()

train_samples = train_samples.astype('float32')
test_samples = test_samples.astype('float32')

TRAIN_BUF = len(train_samples)
BATCH_SIZE = 50
TEST_BUF = len(test_samples)
epochs = 100
latent_dim = 100
num_examples_to_generate = 100
FEATURES_SIZE = 1000

train_dataset = tf.data.Dataset.from_tensor_slices(train_samples).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_samples).shuffle(TEST_BUF).batch(BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam(0.0005)

@tf.function
def compute_loss(model, x, epoch=None):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)

  reconstruction_loss = tf.reduce_mean(tf.square(tf.subtract(x, x_logit)))
  epsilon = 1e-10
  recon_loss = -tf.reduce_sum(
      x * tf.math.log(epsilon + x_logit) +
      (1 - x) * tf.math.log(epsilon + 1 - x_logit),
      axis=1
  )
  #reconstruction_loss = tf.reduce_mean(recon_loss) * FEATURES_SIZE
  #reconstruction_loss = tf.multiply(float(FEATURES_SIZE), tf.keras.losses.binary_crossentropy(x, x_logit))
  kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar))
  #kl_loss = tf.reduce_mean(kl_loss)

  if epoch:
      with writer.as_default():
        tf.summary.scalar('mse', data=reconstruction_loss, step=epoch)
        tf.summary.scalar('KL', data=kl_loss, step=epoch)

  return tf.reduce_sum(reconstruction_loss + kl_loss)

@tf.function
def compute_apply_gradients(model, x, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def training():
  for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
      compute_apply_gradients(model, train_x, optimizer)
    end_time = time.time()

    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
      loss(compute_loss(model, test_x, epoch))
    elbo = -loss.result()

    with writer.as_default():
      tf.summary.scalar('elbo', data=elbo, step=epoch)
      tf.summary.histogram('inference first dense layer', data=model.inference_net.layers[0].trainable_weights[0], step=epoch)
      #tf.summary.histogram('inference second dense layer', data=model.inference_net.layers[1].trainable_weights[0],
      #                     step=epoch)
      #tf.summary.histogram('inference output layer', data=model.inference_net.layers[2].trainable_weights[0],
       #                    step=epoch)
      tf.summary.histogram('generative first dense layer', data=model.generative_net.layers[0].trainable_weights[0],
                           step=epoch)
     # tf.summary.histogram('generative second dense layer', data=model.generative_net.layers[1].trainable_weights[0],
      #                     step=epoch)
      #tf.summary.histogram('generative output layer', data=model.generative_net.layers[2].trainable_weights[0],
     #                      step=epoch)

    print('Epoch: {}, Test set ELBO: {}, '
          'time elapse for current epoch {}'.format(epoch,
                                                    elbo,
                                                    end_time - start_time))
    if epoch == epochs:
        check_reconstruction_and_sampling_fidelity(model)

def check_reconstruction_and_sampling_fidelity(vae_model):
    # get means of original columns based on 100 first rows
    original_means = np.mean(df_values[:100], axis=0)

    mean, logvar = vae_model.encode(df_values[:100])
    z = vae_model.reparameterize(mean, logvar)
    x_decoded = vae_model.decode(z)

    decoded_means = np.mean(x_decoded[:100], axis=0)

    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim])
    predictions = vae_model.sample(random_vector_for_generation)

    sampled_means = np.mean(predictions[:100], axis=0)

    pprint(list(zip(original_means, decoded_means, sampled_means)))

model = VAE(latent_dim, (FEATURES_SIZE,))
training()
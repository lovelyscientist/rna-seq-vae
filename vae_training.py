import tensorflow as tf
from model import VAE
import time
from gtex_loader import get_gtex_dataset, plot_dataset_in_3d_space
import numpy as np
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

logdir = "./logs/run3"
writer = tf.summary.create_file_writer(logdir)

(train_samples, train_y), (test_samples, test_y), df_values, df_columns, y_values = get_gtex_dataset()

train_samples = train_samples.astype('float32')
test_samples = test_samples.astype('float32')

TRAIN_BUF = len(train_samples)
BATCH_SIZE = 50
TEST_BUF = len(test_samples)
epochs = 100
latent_dim = 100
num_examples_to_generate = 100
FEATURES_SIZE = 1000

# .shuffle(TRAIN_BUF).batch(BATCH_SIZE)
train_dataset = tf.data.Dataset.from_tensor_slices(train_samples).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_samples).batch(BATCH_SIZE)
train_y_dataset = tf.data.Dataset.from_tensor_slices(train_y).batch(BATCH_SIZE)
test_y_dataset = tf.data.Dataset.from_tensor_slices(test_y).batch(BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam(0.0005)

@tf.function
def compute_loss(model, x, y, epoch=None):
  mean, logvar = model.encode(x, y)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z, y)

  #reconstruction_loss = tf.reduce_mean(tf.square(tf.subtract(x, x_logit)))
  reconstruction_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, x_logit, from_logits=False))
  kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar))


  if epoch:
      with writer.as_default():
        tf.summary.scalar('mse', data=reconstruction_loss, step=epoch)
        tf.summary.scalar('KL', data=kl_loss, step=epoch)

  return tf.reduce_sum(reconstruction_loss + kl_loss)

@tf.function
def compute_apply_gradients(model, x, y, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x, y)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def training():
  tf.summary.trace_on(graph=True, profiler=True)
  for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_y, train_x in zip(train_y_dataset, train_dataset):
      compute_apply_gradients(model, train_x, train_y, optimizer)
    end_time = time.time()

    loss = tf.keras.metrics.Mean()
    for test_y, test_x in zip(test_y_dataset, test_dataset):
      loss(compute_loss(model, test_x, test_y, epoch))
    elbo = -loss.result()

    with writer.as_default():
      tf.summary.scalar('elbo', data=elbo, step=epoch)
      tf.summary.histogram('inference first dense layer', data=model.inference_net.layers[0].trainable_weights[0], step=epoch)
      tf.summary.histogram('generative first dense layer', data=model.generative_net.layers[0].trainable_weights[0],
                           step=epoch)

    print('Epoch: {}, Test set ELBO: {}, '
          'time elapse for current epoch {}'.format(epoch,
                                                    elbo,
                                                    end_time - start_time))
    if epoch == epochs:
        with writer.as_default():
            tf.summary.trace_export(
                name="my_func_trace",
                step=0,
                profiler_outdir=logdir)
        check_reconstruction_and_sampling_fidelity(model)

def check_reconstruction_and_sampling_fidelity(vae_model):
    # get means of original columns based on 100 first rows
    genes_to_validate = 40
    original_means = np.mean(df_values, axis=0)
    original_vars = np.var(df_values, axis=0)

    mean, logvar = vae_model.encode(df_values, y_values)
    z = vae_model.reparameterize(mean, logvar)

    #plot_dataset_in_3d_space(z, y_values)

    x_decoded = vae_model.decode(z, y_values)

    decoded_means = np.mean(x_decoded, axis=0)
    decoded_vars = np.var(x_decoded, axis=0)

    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim])
    predictions = vae_model.sample(random_vector_for_generation, 24.5)

    sampled_means = np.mean(predictions, axis=0)
    sampled_vars = np.var(predictions, axis=0)

    abs_dif = np.divide(np.sum(np.absolute(df_values - x_decoded), axis=0), df_values.shape[0])
    abs_dif_by_mean = np.divide(np.divide(np.sum(np.absolute(df_values - x_decoded), axis=0), df_values.shape[0]), original_means)

    mean_deviations = np.absolute(original_means - decoded_means)
    print(pd.DataFrame(list(zip(df_columns, mean_deviations)), columns=['Gene', 'Mean Dif']).sort_values(by=['Mean Dif'], ascending=False))

    print(predictions[0][:10])
    print(df_values[5][:10])
    print(x_decoded[5][:10])

    #plot_reconstruction_fidelity(original_means[:genes_to_validate], sampled_means[:genes_to_validate], metric_name='Mean')
    #plot_reconstruction_fidelity(original_vars[:genes_to_validate], sampled_vars[:genes_to_validate], metric_name='Variance')
    #plot_reconstruction_fidelity(abs_dif[:genes_to_validate], metric_name='Absolute Difference (Sum by samples)')
    #plot_reconstruction_fidelity(abs_dif_by_mean[:genes_to_validate], metric_name='Absolute Difference (Sum by samples, Divided by gene Mean)')

    print('Sum of Mean difference by gene: ', np.mean(np.absolute(original_means - decoded_means)))
    print('Sum of Absolute difference by gene: ', np.mean(np.sum(np.absolute(df_values - x_decoded), axis=0) / df_values.shape[0]))
    print('Sum of Absolute difference divided by gene Mean: ', np.mean(abs_dif_by_mean))

def plot_reconstruction_fidelity(original_values, sampled_values=[], metric_name=''):
    n_groups = len(original_values)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    if len(sampled_values) > 0:
        plt.bar(index, original_values, bar_width, alpha=opacity, color='b', label='Original')
        plt.bar(index + bar_width, sampled_values, bar_width, alpha=opacity, color='g', label='Reconstructed')
        plt.title('Original VS Reconstructed ' + metric_name)
        plt.xticks(index + bar_width, list(df_columns)[:n_groups], rotation=90)
        plt.ylabel(metric_name + ' Expression Level (Scaled)')
        plt.legend()
    else:
        plt.bar(index, original_values, bar_width, alpha=opacity, color='b')
        plt.title(metric_name)
        plt.xticks(index, list(df_columns)[:n_groups], rotation=90)
        plt.ylabel('Expression Level (Scaled)')
        plt.legend()

    plt.xlabel('Gene Name')

    plt.tight_layout()
    plt.show()


model = VAE(latent_dim, (FEATURES_SIZE,))
training()
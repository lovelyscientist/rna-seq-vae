Variational Autoencoder for gene expression generation

- `model.py ` - Layers and properties of neural network
- `gtex_loder.py` - Loading gene expression dataset
- `vae_training.py` - Model Training and Testing

Snapshots of TensorBoard scalars:

`ELBO` from epoch 1 to n_epochs

![ELBO](./tensorboard_imgs/elbo.png)

`KL Divergence (latent loss)` from epoch 1 to n_epochs

![KL Divergence (latent loss)](./tensorboard_imgs/kl_divergence.png)

`MSE (reconstruction loss)` from epoch 1 to n_epochs

![MSE (reconstruction loss)](./tensorboard_imgs/mse.png)

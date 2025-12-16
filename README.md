## World Model Implementation and Results

### Architecture

* **Vision - Variational Autoencoder:** A Convolutional VAE was employed to compress raw input frames into a low-dimensional latent vector $z$ of size 32. The input images were downsampled to $48 \times 48$ for computational efficiency. The VAE was trained to reconstruct these observations, effectively learning an abstract spatial representation of the track features (curves, borders, and obstacles).
* **Memory:** While the original World Model utilizes an MDN-RNN to predict future states, a streamlined "Frame Stacking" mechanism was implemented. By concatenating the latent vectors of the current and two previous frames ($z_t, z_{t-1}, z_{t-2}$), the controller had the necessary temporal information without the complexity of training a recurrent neural network.
* **Controller:** The controller is a neural network that maps the stacked latent history to the action space (steering, gas, brake). To improve upon the capacity of a simple linear model, a hidden layer with 64 units and `Tanh` activation was implemented, a modification that significantly improved performance.

### Training Methodology

1.  **Phase 1 (Unsupervised Learning):** 10,000 frames were collected using a random policy and trained the VAE using the Adam optimizer. This phase focused solely on minimizing reconstruction loss and KL divergence, establishing a robust "eye" for the agent before any driving behavior was learned.
2.  **Phase 2 (Evolutionary Control):** With the VAE weights frozen, the controller was trained using the CMA-ES. A population size of 64 and an initial standard deviation ($\sigma$) of 0.2 were employed. To mitigate the high variance of procedurally generated tracks (the "Lucky Lap" problem) any agent achieving a new high score was required to validate its performance over 10 random episodes before being saved.

### Results

The final agent demonstrated stable and robust driving behavior. In the final evaluation over 10 random tracks, the agent achieved an average score of **780**.

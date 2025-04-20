
# GAGAN: Generative Adversarial Network with Genetic Algorithms

GAGAN is an image generation framework that enhances traditional Generative Adversarial Networks (GANs) with a population of discriminators evolved using Genetic Algorithms. The generator creates realistic images from random noise, while the discriminators evolve through selection, crossover, and mutation to better distinguish real from fake images.

---

## Requirements

### Install Dependencies

Ensure Python 3.8+ is installed. Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

---

## Dataset

This implementation uses the **CelebA dataset** for training.

1. Download the CelebA dataset from the [official website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
2. Extract it and place it in the following directory structure:

```
./data/CelebA
```

---

## How to Run

### Training the Model

To train the GAGAN model, run:

```bash
python main.py
```

## Key Features

### Generator

- Generates images from random noise using transposed convolution layers.
- Trained to fool the best discriminator in the population.

### 🛡Discriminator Population

- A population of discriminators trained in parallel.
- Fitness is evaluated based on their ability to classify real vs. fake images.

### Genetic Algorithm

- Crossover and mutation operations applied to discriminators.
- Elite selection preserves the best-performing discriminators each generation.

### Training Stability

- Label smoothing and noise injection are used to enhance stability.
- Gradient clipping prevents exploding gradients during backpropagation.

---

## Output

### Generated Images

- Saved in: `generated_images/`
- Example: `generated_images/epoch_10.png` (images after 10 epochs)

### Loss Plots

- Generator and discriminator loss curves are saved as:
  ```
  gagan_loss_plot.png
  ```

### 💾 Saved Models

- Generator and best discriminator models saved periodically:
  ```
  generator_epoch_X.pth
  discriminator_epoch_X.pth
  ```

---

## ⚙️ Customizing Training

Modify the following hyperparameters in `main.py`:

| Hyperparameter      | Description                          |
|---------------------|--------------------------------------|
| `batch_size`        | Training batch size                  |
| `latent_dim`        | Size of input noise vector           |
| `learning_rate_g`   | Learning rate for generator          |
| `learning_rate_d`   | Learning rate for discriminator      |
| `population_size`   | Number of discriminators             |
| `mutation_rate`     | Probability of mutation in population|
| `elite_ratio`       | Fraction of top discriminators kept  |
| `num_epochs`        | Number of training epochs            |

---

## ✅ Testing

1. Make sure the CelebA dataset is correctly placed in `./data/CelebA`.
2. Run the following command to start training and verify functionality:

```bash
python main.py
```

3. Check the `generated_images/` folder for output.

---


# GAGAN: Hybrid Optimization of GANs Using Genetic Algorithms

## Project Overview
GAGAN (Genetic Algorithm-Guided GAN) is a novel approach that enhances the training of Generative Adversarial Networks (GANs) by integrating Genetic Algorithms (GAs) into the training loop. Unlike traditional GANs where both the generator and discriminator are updated using gradient-based backpropagation, GAGAN evolves the discriminator’s weights using a population-based genetic algorithm while the generator continues to be trained with backpropagation. This hybrid framework improves training stability, addresses mode collapse, and boosts the diversity and quality of generated images.

## Paper Details
- **Title:** GAGAN: Enhancing Image Generation Through Hybrid Optimization of Genetic Algorithms and Deep Convolutional Generative Adversarial Networks
- **Authors:** Despoina Konstantopoulou, Paraskevi Zacharia, Michail Papoutsidakis, Helen C. Leligou, Charalampos Patrikakis
- **Year:** 2024
- **Journal:** Algorithms (ISSN 1999-4893)
- **DOI:** [https://doi.org/10.3390/a17120584](https://doi.org/10.3390/a17120584)

## Key Features
- **Hybrid Optimization**: Combines evolutionary search (via GAs) with standard gradient-based learning.
- **Improved Discriminator Training**: Evolving discriminator weights leads to better generalization and feedback for the generator.
- **Enhanced Image Generation**: Results in more stable training and diverse, realistic outputs.
- **Compatible with Standard Datasets**: Demonstrated on MNIST, CIFAR-10, and CelebA datasets.

## Architecture Summary
- **Generator**: Trained with backpropagation using standard convolutional layers.
- **Discriminator**: Trained with both backpropagation and GA-based evolution.
- **Genetic Algorithm Components**:
  - **Selection**: Chooses top-performing discriminators based on fitness.
  - **Crossover**: Combines weights from two parents.
  - **Mutation**: Applies small perturbations to explore new solutions.
- **Population Management**: Multiple discriminator instances are maintained and evolved over generations.

## Advantages Over Traditional GANs
- Reduced sensitivity to hyperparameter tuning.
- Better resistance to mode collapse.
- Faster convergence and improved training stability.
- Higher-quality outputs on complex datasets.

## Technical Challenges
- High computational and memory requirements due to population-based training.
- Implementation complexity in integrating GA with deep learning frameworks.
- Numerical precision and convergence stability concerns during hybrid training.
- Sensitive hyperparameter tuning required for GA components (mutation rate, crossover strategy, etc.).

## Contributors
- **Basil Ali Khan**
- **Ahsan Siddiqui**

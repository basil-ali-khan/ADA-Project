"""def train_epoch(self, dataloader, num_generations):
        
        g_losses = []
        d_losses = []
        
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)
            
            # Generate fake images
            noise = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_images = self.generator(noise)
            
            # For each generation in this batch
            for gen in range(num_generations):
                # Evaluate fitness of each discriminator
                for j, discriminator in enumerate(self.discriminator_population):
                    self.fitness_scores[j] = self.evaluate_discriminator_fitness(
                        discriminator, real_images, fake_images)
                
                # Train each discriminator using gradient-based method first
                avg_d_loss = 0
                for j, (discriminator, optimizer) in enumerate(zip(
                        self.discriminator_population, self.discriminator_optimizers)):
                    d_loss, real_score, fake_score = self.train_discriminator_grad_based(
                        discriminator, optimizer, real_images, fake_images)
                    avg_d_loss += d_loss
                
                avg_d_loss /= self.population_size
                d_losses.append(avg_d_loss)
                
                # Perform genetic evolution
                self.genetic_evolution(elite_ratio, mutation_rate, mutation_strength)
                
                # Re-evaluate fitness to find the best discriminator
                for j, discriminator in enumerate(self.discriminator_population):
                    self.fitness_scores[j] = self.evaluate_discriminator_fitness(
                        discriminator, real_images, fake_images)
                    
                best_idx = np.argmax(self.fitness_scores)
                best_discriminator = self.discriminator_population[best_idx]
                
                # Generate new fake images for training the generator
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_images = self.generator(noise)
                
                # Train the generator with the best discriminator
                g_loss = self.train_generator(fake_images, best_discriminator)
                g_losses.append(g_loss)
                
                # Print progress
                if i % 50 == 0 and gen == num_generations - 1:
                    print(f"Batch {i}/{len(dataloader)}, Gen {gen+1}/{num_generations}, "
                          f"D Loss: {avg_d_loss:.4f}, G Loss: {g_loss:.4f}")
        
        # Return average losses for the epoch
        return np.mean(g_losses), np.mean(d_losses)"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import copy
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os
from torch.optim.lr_scheduler import ExponentialLR

# Create output directory if not exists
os.makedirs('generated_images', exist_ok=True)


# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Hyperparameters
batch_size = 32
latent_dim = 100
image_channels = 3  # 3 for RGB, 1 for grayscale
hidden_channels = 64
# Modify hyperparameters
learning_rate_d = 0.00005  # Significantly lower learning rate for discriminator
learning_rate_g = 0.0001   # Generator learning rate
lr_decay = 0.99           # Learning rate decay factor
beta1 = 0.5
num_epochs = 50
population_size = 10  # Number of discriminators in population
elite_ratio = 0.3     # Proportion of population to keep for crossover
mutation_rate = 0.1   # Probability of mutation for each weight
mutation_strength = 0.1  # How much to mutate weights
num_generations = 5   # Number of GA generations per epoch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define generator network
class Generator(nn.Module):
    def __init__(self, latent_dim, image_channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is latent_dim, going into a convolution
            nn.ConvTranspose2d(latent_dim, hidden_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_channels * 8),
            nn.ReLU(True),
            # state size: (hidden_channels*8) x 4 x 4
            nn.ConvTranspose2d(hidden_channels * 8, hidden_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(True),
            # state size: (hidden_channels*4) x 8 x 8
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(True),
            # state size: (hidden_channels*2) x 16 x 16
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            # state size: (hidden_channels) x 32 x 32
            nn.ConvTranspose2d(hidden_channels, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: (image_channels) x 64 x 64
        )

    def forward(self, x):
        # x is of shape (batch_size, latent_dim)
        x = x.view(-1, latent_dim, 1, 1)
        return self.main(x)

# Define discriminator network
class Discriminator(nn.Module):
    def __init__(self, image_channels):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (image_channels) x 64 x 64
            nn.Conv2d(image_channels, hidden_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (hidden_channels) x 32 x 32
            nn.Conv2d(hidden_channels, hidden_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (hidden_channels*2) x 16 x 16
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (hidden_channels*4) x 8 x 8
            nn.Conv2d(hidden_channels * 4, hidden_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (hidden_channels*8) x 4 x 4
            nn.Conv2d(hidden_channels * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)

# GAGAN model implementation
class GAGAN:
    def __init__(self, latent_dim, image_channels, population_size, device):
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.population_size = population_size
        self.device = device
        self.fixed_noise = torch.randn(64, self.latent_dim, device=self.device)
        
        # Initialize generator and discriminator models
        self.generator = Generator(latent_dim, image_channels).to(device)
        
        # Initialize a population of discriminators
        self.discriminator_population = [Discriminator(image_channels).to(device) for _ in range(population_size)]
        
        # Fitness scores for discriminators
        self.fitness_scores = [0] * population_size
        
        # Lower learning rate for discriminator - CRITICAL for stability
        discriminator_lr = 0.00001  # Much lower than generator learning rate
        generator_lr = 0.0001
        
        # Initialize optimizers AFTER creating the models
        self.generator_optimizer = optim.Adam(self.generator.parameters(), 
                                        lr=generator_lr, betas=(0.5, 0.999))
        self.discriminator_optimizers = [optim.Adam(disc.parameters(), 
                                            lr=discriminator_lr, betas=(0.5, 0.999)) 
                                    for disc in self.discriminator_population]
        
        # Learning rate schedulers with gentler decay
        self.generator_scheduler = ExponentialLR(self.generator_optimizer, gamma=0.995)
        self.discriminator_schedulers = [ExponentialLR(opt, gamma=0.995) 
                                for opt in self.discriminator_optimizers]
        
        # Loss function
        self.criterion = nn.BCELoss()
    
    def train_discriminator_with_smoothing(self, discriminator, optimizer, real_images, fake_images):
        """Train discriminator with one-sided label smoothing"""
        batch_size = real_images.size(0)
        
        # Labels with one-sided smoothing (only smooth real labels)
        real_labels = torch.full((batch_size,), 0.9, device=self.device)  # Smoothed from 1.0
        fake_labels = torch.full((batch_size,), 0.0, device=self.device)  # Keep at 0
        
        # Add noise to input images (helps with stability)
        real_images_noisy = self.add_noise_to_images(real_images, noise_factor=0.01)
        fake_images_noisy = self.add_noise_to_images(fake_images.detach(), noise_factor=0.01)
        
        # Train with real images
        optimizer.zero_grad()
        real_outputs = discriminator(real_images_noisy)
        real_loss = self.criterion(real_outputs, real_labels)
        
        # Train with fake images
        fake_outputs = discriminator(fake_images_noisy)  # Detach to avoid training G
        fake_loss = self.criterion(fake_outputs, fake_labels)
        
        # Combined loss
        d_loss = real_loss + fake_loss
        
        # Gradient penalty (simplified Wasserstein-GP inspired approach)
        # This restricts the discriminator from becoming too powerful
        for param in discriminator.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-0.01, 0.01)
        
        d_loss.backward()
        
        # Clip gradients (important for stability)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Return metrics
        real_score = real_outputs.mean().item()
        fake_score = fake_outputs.mean().item()
        
        return d_loss.item(), real_score, fake_score

    def add_noise_to_images(self, images, noise_factor=0.05):
        """Add small random noise to images to improve training stability"""
        noise = torch.randn_like(images) * noise_factor
        noisy_images = images + noise
        # Clamp values to valid range [-1, 1] for tanh-normalized images
        return torch.clamp(noisy_images, -1, 1)
        
    def train_discriminator_grad_based(self, discriminator, optimizer, real_images, fake_images):
        """Train a discriminator using gradient-based method (standard backpropagation)"""
        batch_size = real_images.size(0)
        
        # Labels
        real_labels = torch.full((batch_size,), 1.0, device=self.device)
        fake_labels = torch.full((batch_size,), 0.0, device=self.device)
        
        # Train with real images
        optimizer.zero_grad()
        real_outputs = discriminator(real_images)
        real_loss = self.criterion(real_outputs, real_labels)
        
        # Train with fake images
        fake_outputs = discriminator(fake_images.detach())  # Detach to avoid training G
        fake_loss = self.criterion(fake_outputs, fake_labels)
        
        # Combined loss and optimize
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer.step()
        
        # Return metrics
        real_score = real_outputs.mean().item()
        fake_score = fake_outputs.mean().item()
        
        return d_loss.item(), real_score, fake_score
    
    def select_elite_discriminators(self, elite_ratio):
        """Select the top-performing discriminators based on fitness scores"""
        elite_count = max(1, int(self.population_size * elite_ratio))
        sorted_indices = np.argsort(self.fitness_scores)[::-1]  # Sort by fitness (descending)
        elite_indices = sorted_indices[:elite_count]
        
        return elite_indices
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parent discriminators to create a child"""
        child = Discriminator(self.image_channels).to(self.device)
        
        # Get state dicts
        parent1_dict = parent1.state_dict()
        parent2_dict = parent2.state_dict()
        child_dict = child.state_dict()
        
        # For each parameter, randomly choose from either parent
        for key in child_dict:
            # Randomly select which parent to inherit from for this parameter
            if random.random() < 0.5:
                child_dict[key] = parent1_dict[key].clone()
            else:
                child_dict[key] = parent2_dict[key].clone()
        
        # Load the new state dict into the child
        child.load_state_dict(child_dict)
        
        return child
    
    def mutate(self, discriminator, mutation_rate, mutation_strength):
        """Apply mutation to a discriminator's weights"""
        state_dict = discriminator.state_dict()
        
        for key in state_dict:
            # Only mutate weights, not biases or other parameters
            if 'weight' in key:
                mask = torch.rand_like(state_dict[key]) < mutation_rate
                mutation = torch.randn_like(state_dict[key]) * mutation_strength
                state_dict[key] = torch.where(mask, state_dict[key] + mutation, state_dict[key])
        
        discriminator.load_state_dict(state_dict)
        
        return discriminator
    

    def train_generator(self, fake_images, best_discriminator):
        """Train generator with improved stability"""
        batch_size = fake_images.size(0)
        
        # Labels - we want the generator to make images that the discriminator thinks are real
        # Use 1.0 (not smoothed) for generator training
        real_labels = torch.full((batch_size,), 1.0, device=self.device)
        
        # Zero generator gradients
        self.generator_optimizer.zero_grad()
        
        # Forward pass of fake images through discriminator
        outputs = best_discriminator(fake_images)
        
        # Calculate G's loss based on this output
        g_loss = self.criterion(outputs, real_labels)
        
        # Calculate gradients for G
        g_loss.backward()
        
        # Clip gradients to prevent extreme updates
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        
        # Update G
        self.generator_optimizer.step()
        
        return g_loss.item()

    def train_epoch(self, dataloader, num_generations):
        """Simplified and more stable training loop"""
        g_losses = []
        d_losses = []
        max_batches = min(20, len(dataloader))  # Limit batches for testing

        for i, (real_images, _) in enumerate(dataloader):
            if i >= max_batches:
                break

            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)
            
            # ===== Phase 1: Train Discriminator =====
            # Train discriminator less frequently (crucial for stability)
            if i % 2 == 0:  # Train D only every other batch
                # Generate fake images (detached)
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                with torch.no_grad():  # Don't accumulate gradients for G
                    fake_images = self.generator(noise)
                
                # Train discriminators
                avg_d_loss = 0
                for j, (discriminator, optimizer) in enumerate(zip(
                        self.discriminator_population, self.discriminator_optimizers)):
                    d_loss, real_score, fake_score = self.train_discriminator_with_smoothing(
                        discriminator, optimizer, real_images, fake_images)
                    avg_d_loss += d_loss
                
                avg_d_loss /= self.population_size
                d_losses.append(avg_d_loss)
                
                # Perform genetic evolution only every few batches
                if i % 4 == 0:
                    self.genetic_evolution(elite_ratio=0.3, 
                                          mutation_rate=0.05,  # Lower mutation rate
                                          mutation_strength=0.05)  # Lower mutation strength
            
            # ===== Phase 2: Train Generator =====
            # Evaluate fitness to find best discriminator
            noise = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_images = self.generator(noise)
            
            for j, discriminator in enumerate(self.discriminator_population):
                with torch.no_grad():  # Calculate fitness without gradients
                    self.fitness_scores[j] = self.evaluate_discriminator_fitness(
                        discriminator, real_images, fake_images)
            
            best_idx = np.argmax(self.fitness_scores)
            best_discriminator = self.discriminator_population[best_idx]
            
            # Train generator with the best discriminator
            # Train G more frequently than D (key to preventing D from overwhelming G)
            g_loss = self.train_generator(fake_images, best_discriminator)
            g_losses.append(g_loss)
            
            if i % 5 == 0:
                real_score = best_discriminator(real_images).mean().item()
                fake_score = best_discriminator(fake_images).mean().item()
                
                # Format the D Loss differently depending on whether it exists
                if i % 2 == 0:
                    d_loss_str = f"{avg_d_loss:.4f}"
                else:
                    d_loss_str = "N/A"
                    
                print(f"Batch {i}/{len(dataloader)}, "
                    f"D Loss: {d_loss_str}, "
                    f"G Loss: {g_loss:.4f}, "
                    f"Real Score: {real_score:.4f}, Fake Score: {fake_score:.4f}")

        # Update learning rates at the end of each epoch
        self.generator_scheduler.step()
        for scheduler in self.discriminator_schedulers:
            scheduler.step()

        return np.mean(g_losses) if g_losses else 0.0, np.mean(d_losses) if d_losses else 0.0

    
    def generate_samples(self, num_samples):
        """Generate sample images using the trained generator"""
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim, device=self.device)
            generated_images = self.generator(noise)
            
        return generated_images
    
    def save_models(self, generator_path, discriminator_path):
        """Save the trained models"""
        torch.save(self.generator.state_dict(), generator_path)
        
        # Save the best discriminator (highest fitness)
        best_idx = np.argmax(self.fitness_scores)
        best_discriminator = self.discriminator_population[best_idx]
        torch.save(best_discriminator.state_dict(), discriminator_path)
    
    def load_models(self, generator_path, discriminator_path):
        """Load pre-trained models"""
        self.generator.load_state_dict(torch.load(generator_path))
        
        # Load one discriminator
        self.discriminator_population[0].load_state_dict(torch.load(discriminator_path))
        
        # Clone this discriminator for the entire population
        for i in range(1, self.population_size):
            self.discriminator_population[i].load_state_dict(
                self.discriminator_population[0].state_dict())
            
            # Add some mutations to ensure diversity
            self.discriminator_population[i] = self.mutate(
                self.discriminator_population[i], mutation_rate, mutation_strength)
    
    def evaluate_discriminator_fitness(self, discriminator, real_images, fake_images):
        """Improved fitness evaluation with regularization"""
        batch_size = real_images.size(0)
        
        # Labels
        real_labels = torch.full((batch_size,), 1.0, device=self.device)
        fake_labels = torch.full((batch_size,), 0.0, device=self.device)
        
        # Forward pass - real images
        real_outputs = discriminator(real_images)
        real_loss = self.criterion(real_outputs, real_labels)
        
        # Forward pass - fake images
        fake_outputs = discriminator(fake_images.detach())
        fake_loss = self.criterion(fake_outputs, fake_labels)
        
        d_loss = real_loss + fake_loss
        
        # Calculate accuracy as another fitness metric
        real_accuracy = ((real_outputs > 0.5).float() == real_labels).float().mean().item()
        fake_accuracy = ((fake_outputs < 0.5).float() == (1 - fake_labels)).float().mean().item()
        accuracy = (real_accuracy + fake_accuracy) / 2
        
        # Calculate Wasserstein-like estimate (real_score - fake_score)
        wasserstein_estimate = real_outputs.mean().item() - fake_outputs.mean().item()
        
        # Penalize extreme discriminators (if the gap is too large, it's probably overfitting)
        regularization = -abs(wasserstein_estimate - 0.5) * 2.0
        
        # Combine metrics for fitness (higher is better)
        # Balance accuracy with moderate Wasserstein distance
        fitness = accuracy + 0.2 * regularization - 0.5 * d_loss.item()
        
        return fitness

    def genetic_evolution(self, elite_ratio, mutation_rate, mutation_strength):
        """Modified genetic evolution to maintain diversity"""
        # Select elite discriminators
        elite_indices = self.select_elite_discriminators(elite_ratio)
        elite_discriminators = [copy.deepcopy(self.discriminator_population[i]) for i in elite_indices]
        
        # Create new population starting with elites
        new_population = copy.deepcopy(elite_discriminators)
        
        # Fill the rest of the population with crossover and mutation
        while len(new_population) < self.population_size:
            # Select parents with tournament selection
            parent1 = random.choice(elite_discriminators)
            parent2 = random.choice(elite_discriminators)
            
            # Create child through crossover
            child = self.crossover(parent1, parent2)
            
            # Apply mutation with reduced rate and strength
            child = self.mutate(child, mutation_rate, mutation_strength)
            
            # Add to new population
            new_population.append(child)
        
        # Replace the old population
        self.discriminator_population = new_population
        
        # Create new optimizers for the new population with the same learning rate
        discriminator_lr = self.discriminator_optimizers[0].param_groups[0]['lr']
        self.discriminator_optimizers = [optim.Adam(disc.parameters(), 
                                      lr=discriminator_lr, betas=(0.5, 0.999)) 
                                  for disc in self.discriminator_population]

    def save_generated_images(self, epoch):
        """Generate and save images using fixed noise."""
        self.generator.eval()
        with torch.no_grad():
            fake_images = self.generator(self.fixed_noise).detach().cpu()
            # Ensure directory exists
            os.makedirs('generated_images', exist_ok=True)
            # Save grid of images
            vutils.save_image(fake_images, f"generated_images/epoch_{epoch}.png",
                            normalize=True, nrow=8)
            
            # Optional display with matplotlib
            try:
                grid = vutils.make_grid(fake_images, padding=2, normalize=True)
                npimg = grid.numpy()
                plt.figure(figsize=(8, 8))
                plt.axis("off")
                plt.title(f"Generated Images at Epoch {epoch}")
                plt.imshow(np.transpose(npimg, (1, 2, 0)))
                plt.savefig(f"generated_images/plot_epoch_{epoch}.png")
                plt.close()  # Close to prevent memory leaks
            except Exception as e:
                print(f"Visualization error: {e}")
                
        self.generator.train()



# Main training function
def train_gagan(dataset_path, image_size=64, num_epochs=50, save_interval=5, batch_size=32, num_generations=5):
    """Train the GAGAN model on a dataset"""
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Load dataset
    dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)
    
    # Limit to 10% of the dataset
    subset_size = int(len(dataset) * 0.1)
    subset = Subset(dataset, range(subset_size))
    
    # Create DataLoader for the subset
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize GAGAN
    model = GAGAN(latent_dim, image_channels, population_size, device)
    
    # Lists to store losses
    g_losses = []
    d_losses = []
    
    # Lists to store generated images for visualization
    fixed_noise = torch.randn(64, latent_dim, device=device)
    img_list = []
    
    print("Starting Training...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train one epoch
        g_loss, d_loss = model.train_epoch(dataloader, num_generations)
        model.save_generated_images(epoch)
        
        # Record losses
        g_losses.append(g_loss)
        d_losses.append(d_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")
        
        # Save sample generated images
        if epoch % save_interval == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                fake_images = model.generator(fixed_noise).detach().cpu()
            img_list.append(torchvision.utils.make_grid(fake_images, padding=2, normalize=True))
            
            # Save models
            model.save_models(f"generator_epoch_{epoch+1}.pth", f"discriminator_epoch_{epoch+1}.pth")
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("gagan_loss_plot.png")
    
    # Create animation of generated images throughout training
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    plt.savefig("gagan_generated_images.png")
    
    return model, g_losses, d_losses, img_list

if __name__ == "__main__":
    image_size = 64
    batch_size = 32
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CelebA dataset
    dataset_path = './data/CelebA'  # Update the dataset path here
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    # Initialize and train the GAGAN model
    model, g_losses, d_losses, img_list = train_gagan(dataset_path, image_size=image_size, num_epochs=num_epochs, save_interval=5)
    
    # Save final models
    torch.save(model.generator.state_dict(), "generator_final.pth")
    torch.save(model.population[0].state_dict(), "discriminator_final.pth")

    print("Training complete!")

    # Generate final sample grid
    with torch.no_grad():
        z = torch.randn(16, latent_dim, device=device)
        samples = model.generator(z).detach().cpu()

    grid = torchvision.utils.make_grid(samples, padding=2, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Final Generated Images")
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.savefig("final_generated_images.png")
    plt.show()


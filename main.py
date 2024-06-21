import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


# Define the CNN architecture
class CharacterClassifier(nn.Module):
    def __init__(self):
        super(CharacterClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Load and preprocess the MNIST dataset
train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)

# Instantiate the model, loss function, and optimizer
model = CharacterClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store the loss and accuracy values for plotting
train_losses = []
train_accuracies = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += torch.sum(torch.eq(labels, predicted)).item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%"
    )
# Save the trained model with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"character_classifier_{timestamp}.pth"
torch.save(model.state_dict(), model_path)
print(f"Trained model saved as: {model_path}")

# Plot the training curve
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()


# Function to add salt and pepper noise to images
def add_salt_and_pepper_noise(images, noise_prob):
    noisy_images = images.clone()
    mask = torch.rand_like(images) < noise_prob
    noisy_images[mask] = 1 - images[mask]
    return noisy_images


# Noise probabilities
noise_probs = [0.1, 0.2, 0.3, 0.5]

# Lists to store the accuracies for different noise probabilities
accuracies = []

# Evaluation with different noise probabilities
for noise_prob in noise_probs:
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Add salt and pepper noise to the images
            noisy_images = add_salt_and_pepper_noise(images, noise_prob)

            outputs = model(noisy_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(torch.eq(labels, predicted)).item()

        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f"Noise Probability: {noise_prob}, Accuracy: {accuracy:.2f}%")

# Plot the accuracies for different noise probabilities
plt.figure(figsize=(8, 4))
plt.plot(noise_probs, accuracies, marker="o")
plt.xlabel("Noise Probability")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs. Noise Probability")
plt.grid(True)
plt.show()

# Select a random image from the test dataset
index = np.random.randint(len(test_dataset))
image, label = test_dataset[index]
image = image.unsqueeze(0).to(device)

# Display the original image and noisy images for each noise probability
fig, axes = plt.subplots(1, len(noise_probs) + 1, figsize=(16, 2))
axes[0].imshow(image.squeeze().cpu().numpy(), cmap="gray")
axes[0].set_title(f"Original (Label: {label})")
axes[0].axis("off")

for i, noise_prob in enumerate(noise_probs):
    noisy_image = add_salt_and_pepper_noise(image, noise_prob)
    axes[i + 1].imshow(noisy_image.squeeze().cpu().numpy(), cmap="gray")
    axes[i + 1].set_title(f"Noise Prob: {noise_prob}")
    axes[i + 1].axis("off")

plt.tight_layout()
plt.show()

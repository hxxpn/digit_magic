import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image, ImageOps
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import os

app = Flask(__name__)


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
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = CharacterClassifier()
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)


def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')

    # Invert colors (MNIST has white digits on black background)
    image = ImageOps.invert(image)

    # Resize to 28x28
    image = image.resize((28, 28), Image.LANCZOS)

    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0

    # Center the digit
    center_of_mass = np.array(img_array > 0.5).nonzero()
    top, left = np.min(center_of_mass, axis=1)
    bottom, right = np.max(center_of_mass, axis=1)
    center = ((top + bottom) // 2, (left + right) // 2)
    shift = np.array([14, 14]) - np.array(center)
    img_array = np.roll(img_array, shift, axis=(0, 1))

    # Save the processed image for debugging
    Image.fromarray((img_array * 255).astype(np.uint8)).save('debug_processed_image.png')

    # Convert to tensor
    image_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)

    return image_tensor


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    image_data = request.json['imageData']
    image_data = image_data.split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    image_tensor = preprocess_image(image)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor.to(device))
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
        result = predicted.item()

    return jsonify({'result': result, 'probabilities': probabilities[0].cpu().tolist()})


def train_model():
    global model

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()

    # Create 'static' directory if it doesn't exist
    os.makedirs('static', exist_ok=True)

    plt.savefig('static/training_results.png')

    train_model.results = {'loss': losses[-1], 'accuracy': accuracies[-1]}

    # Save the model
    torch.save(model.state_dict(), 'static/mnist_model.pth')


@app.route('/train', methods=['POST'])
def start_training():
    thread = Thread(target=train_model)
    thread.start()
    return jsonify({'message': 'Training started'})


@app.route('/training_status', methods=['GET'])
def training_status():
    if hasattr(train_model, 'results'):
        return jsonify({'status': 'completed', 'results': train_model.results})
    else:
        return jsonify({'status': 'in_progress'})


if __name__ == '__main__':
    # Load the model if it exists
    if os.path.exists('static/mnist_model.pth'):
        model.load_state_dict(torch.load('static/mnist_model.pth', map_location=device))
        print("Loaded existing model")
    app.run(debug=True)
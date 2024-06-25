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
import random

app = Flask(__name__)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
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


class JednostavnaMreza(nn.Module):
    def __init__(self):
        super(JednostavnaMreza, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.fc3(x)
        return x


models = {
    'cnn': ConvNet(),
    'mlp': JednostavnaMreza()
}

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
for model in models.values():
    model.to(device)

current_model = 'cnn'


def add_salt_and_pepper_noise(image, prob):
    output = np.copy(image)
    noise = np.random.random(output.shape)
    output[noise < prob / 2] = 0
    output[noise > 1 - prob / 2] = 1
    return output


def preprocess_image(image):
    image = image.convert('L')
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    img_array = np.array(image).astype(np.float32) / 255.0
    return img_array


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    global current_model
    image_data = request.json['imageData']
    model_type = request.json['modelType']
    current_model = model_type

    image_data = image_data.split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    image_tensor = preprocess_image(image)
    image_tensor = torch.FloatTensor(image_tensor).unsqueeze(0).unsqueeze(0)

    model = models[current_model]
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.to(device))
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
        result = predicted.item()

    return jsonify({'result': result, 'probabilities': probabilities[0].cpu().tolist()})


@app.route('/classify_with_noise', methods=['POST'])
def classify_with_noise():
    global current_model
    image_data = request.json['imageData']
    model_type = request.json['modelType']
    current_model = model_type

    image_data = image_data.split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    img_array = preprocess_image(image)

    noise_probs = [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = []

    for prob in noise_probs:
        noisy_img = add_salt_and_pepper_noise(img_array, prob)
        noisy_tensor = torch.FloatTensor(noisy_img).unsqueeze(0).unsqueeze(0)

        model = models[current_model]
        model.eval()
        with torch.no_grad():
            output = model(noisy_tensor.to(device))
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            result = predicted.item()

        display_img = (1 - noisy_img) * 255
        noisy_image = Image.fromarray(display_img.astype(np.uint8))
        buffered = io.BytesIO()
        noisy_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        results.append({
            'noise_prob': prob,
            'result': result,
            'probabilities': probabilities[0].cpu().tolist(),
            'image': f'data:image/png;base64,{img_str}'
        })

    return jsonify(results)


@app.route('/test_mnist_subset', methods=['POST'])
def test_mnist_subset():
    model_type = request.json['modelType']

    # Load MNIST test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Select random subset
    subset_size = 2000
    subset_indices = random.sample(range(len(test_dataset)), subset_size)
    subset = torch.utils.data.Subset(test_dataset, subset_indices)

    model = models[model_type]
    model.eval()

    noise_probs = [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
    accuracies = []

    for prob in noise_probs:
        correct = 0
        total = 0

        for image, label in subset:
            noisy_img = add_salt_and_pepper_noise(image.squeeze().numpy(), prob)
            noisy_tensor = torch.FloatTensor(noisy_img).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                output = model(noisy_tensor.to(device))
                _, predicted = torch.max(output, 1)
                total += 1
                correct += (predicted.item() == label)

        accuracy = 100 * correct / total
        accuracies.append(accuracy)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(noise_probs, accuracies, marker='o')
    plt.title(f'Tacnost u odnosu na verovatnocu suma {model_type.upper()}')
    plt.xlabel('Verovatnoca suma')
    plt.ylabel('Tacnost (%)')
    plt.grid(True)

    # Save plot
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_str = base64.b64encode(img_buf.getvalue()).decode()

    return jsonify({
        'accuracies': accuracies,
        'plot': f'data:image/png;base64,{img_str}'
    })


def train_model(model_type):
    global models

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    model = models[model_type]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 15
    losses = []
    accuracies = []

    try:
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

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title('Gubitak tokom obuke')
        plt.xlabel('Epoha')
        plt.ylabel('Gubitak')

        plt.subplot(1, 2, 2)
        plt.plot(accuracies)
        plt.title('Tacnost tokom obuke')
        plt.xlabel('Epoha')
        plt.ylabel('Tacnost (%)')

        plt.tight_layout()

        os.makedirs('static', exist_ok=True)

        plt.savefig(f'static/training_results_{model_type}.png')

        train_model.results = {'loss': losses[-1], 'accuracy': accuracies[-1]}

        torch.save(model.state_dict(), f'static/mnist_model_{model_type}.pth')

        print(f"Trening zavrsen za {model_type} model")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        train_model.results = {'error': str(e)}


@app.route('/train', methods=['POST'])
def start_training():
    model_type = request.json['modelType']
    thread = Thread(target=train_model, args=(model_type,))
    thread.start()
    return jsonify({'message': f'Trening zapocet za {model_type}'})


@app.route('/training_status', methods=['GET'])
def training_status():
    if hasattr(train_model, 'results'):
        return jsonify({'status': 'completed', 'results': train_model.results})
    else:
        return jsonify({'status': 'in_progress'})


if __name__ == '__main__':
    for model_type in models:
        if os.path.exists(f'static/mnist_model_{model_type}.pth'):
            models[model_type].load_state_dict(torch.load(f'static/mnist_model_{model_type}.pth', map_location=device))
            print(f"Ucitan postojeci model: {model_type}")
    app.run(debug=True)

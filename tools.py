def online_rlhf():
    pygame.init()
    total_images = 0
    correct_predictions = 0
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((500, 500)),
        transforms.ToTensor()
    ])
    model = ImageClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    running_loss = 0
    for image in os.listdir(image_dir):
        labeled = False
        image_path = os.path.join(image_dir, image)
        image = Image.open(image_path)
        image = transform(image)
        image_surface = pygame.image.load(image_path)
        screen = pygame.display.set_mode((1000, 900))
        screen.blit(image_surface, (0, 0))
        pygame.display.flip()
        while not labeled:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        labels = torch.tensor([1])
                        labeled = True
                        continue
                    elif event.key == pygame.K_RIGHT:
                        labels = torch.tensor([0])
                        labeled = True
                        continue
                    elif event.key == pygame.K_SPACE:
                        labels = None
                        labeled = True
                        continue
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return model
        optimizer.zero_grad()
        outputs = model(image.unsqueeze(0))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_images += 1
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)
            if predicted == labels:
                correct_predictions += 1
        accuracy = correct_predictions / total_images if total_images > 0 else 0
        average_loss = running_loss / total_images if total_images > 0 else 0
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Average Loss: {average_loss:.4f}")


        class ImageClassifier(nn.Module):
        def __init__(self):
            super(ImageClassifier, self).__init__()
            self.conv1 = nn.Conv2d(1, 3, 3)
            self.conv2 = nn.Conv2d(3, 6, 3)
            self.conv3 = nn.Conv2d(6, 9, 3)
            self.pool = nn.MaxPool2d(3, 3)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(2601, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 2)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            x = self.flatten(x)
            x = self.dropout(x)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x


import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image


one_labels_dir = "/home/cam/testing/trueims"
zero_labels_dir = "/home/cam/testing/falseims"

def load_and_preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image_extension = tf.strings.split(image_path, '.')[-1]
    if tf.strings.lower(image_extension) == 'png':
        image = tf.image.decode_png(image, channels=1)
    elif tf.strings.lower(image_extension) == 'jpg' or tf.strings.lower(image_extension) == 'jpeg':
        image = tf.image.decode_jpeg(image, channels=1)
    else:
        raise ValueError(f"Unsupported image format: {image_extension}")
    image = tf.image.resize(image, [500, 500])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Get the list of image file paths in the "true" folder
true_image_paths = [os.path.join(one_labels_dir, filename) for filename in os.listdir(one_labels_dir) if os.path.isfile(os.path.join(one_labels_dir, filename))]

true_dataset = tf.data.Dataset.from_generator(
    lambda: (load_and_preprocess_image(image_path, 1) for image_path in true_image_paths),
    output_signature=(tf.TensorSpec(shape=(500, 500, 1), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int32))
)

# Get the list of image file paths in the "false" folder
false_image_paths = [os.path.join(zero_labels_dir, filename) for filename in os.listdir(zero_labels_dir) if os.path.isfile(os.path.join(zero_labels_dir, filename))]

false_dataset = tf.data.Dataset.from_generator(
    lambda: (load_and_preprocess_image(image_path, 0) for image_path in false_image_paths),
    output_signature=(tf.TensorSpec(shape=(500, 500, 1), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int32))
)

# Merge the true and false datasets
dataset = true_dataset.concatenate(false_dataset)

# Calculate the length of the dataset
dataset_length = len(list(dataset))

# Shuffle the dataset
dataset = dataset.shuffle(buffer_size=dataset_length)

# Save the shuffled dataset as a TensorFlow dataset
tf.data.Dataset.save(dataset, 'chat_dataset')
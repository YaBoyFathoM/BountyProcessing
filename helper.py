import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
import torch.nn as nn
import h5py
#My dataset can be found at https://drive.google.com/drive/folders/18MNN1IwYqFeSQ7AlLg091oZ9G863EEfH?usp=drive_link
json_path = 'segment_data/result.json'
images_path = 'segment_data/'




#this is a dataset of 824 labled images
#the two bbox classes are chatbot and human
#ideally I want to crop and categorize them seperately so the human side can be used to fine tune our LLM.

#Im building the infrastructure for the LLM rn, but I want to enable the use of screenshots for convenience on mobile
def demo():
    transform = ToTensor()
    coco_dataset = CocoDetection(root=images_path, annFile=json_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(coco_dataset, batch_size=1, shuffle=True)
    category_colors = {
        0: 'blue',
        1: 'green',
    }
    category_labels = {
        0: 'chatbot',
        1: 'human',
    }
    fig, ax = plt.subplots(1)
    count = 0
    for image, targets in data_loader:
        image = image.squeeze(0)
        image = image.permute(1, 2, 0)
        image = image.numpy()
        ax.imshow(image)
        for target in targets:
            xmin = target['bbox'][0].item()
            ymin = target['bbox'][1].item()
            xmax = (target['bbox'][0] + target['bbox'][2]).item()
            ymax = (target['bbox'][1] + target['bbox'][3]).item()
            category_id = target['category_id'].item()
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor=category_colors[category_id], facecolor='none')
            ax.add_patch(rect)
            label = category_labels[category_id]
            ax.text(xmin, ymin, label, color='white', fontsize=8, verticalalignment='top', bbox={'color': category_colors[category_id], 'pad': 0})
        plt.pause(2)
        ax.clear()
        count += 1
        if count == 5:
            plt.close()
            break
    plt.show()
demo()

#in the event you want to pull more data,this is a cnn that does fairly well at rejecting non-chatbot screenshots
#1=good 0=bad you know the deal
class TextImageClassifier(nn.Module):
    def __init__(self):
        super(TextImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.conv4(x)
        x = nn.ReLU()(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    def load():
        chatcleaner_cnn = h5py.File('chat_cleaner.h5', 'r')
        model = TextImageClassifier()
        for name, param in model.named_parameters():
            if name in chatcleaner_cnn:
                weight = torch.from_numpy(chatcleaner_cnn[name][:])
                param.data.copy_(weight)
        chatcleaner_cnn.close()
        return model
Model=TextImageClassifier.load()


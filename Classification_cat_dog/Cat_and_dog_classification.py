#%% 
import numpy as np
from collections import OrderedDict
import torch
from torch import optim
import torch.nn as nn
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# %% Data preparation

train_dir = 'train'
test_dir = 'test'

transform = transforms.Compose([transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor()])

import torch
torch.manual_seed(10)

train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

#%% show some images
def imshow(image_torch):
    # flip image channels to RGB
    image_torch = image_torch.numpy().transpose((1, 2, 0))
    plt.figure()
    plt.imshow(image_torch)

X_train, y_train = next(iter(train_loader))

# make a grid from a batch
image_grid = torchvision.utils.make_grid(X_train[:16, :, :, :], scale_each= True, nrow=4)

imshow(image_grid)

#%%
# Download and instantiate pre-trained Densenet
model = models.densenet121(weights='DenseNet121_Weights.DEFAULT')

#%% Freeze all layers
for params in model.parameters():
    params.requires_grad = False

#for params in model.features[-1:].parameters():
#     params.requires_grad = True

#model.features[-1].weight.requires_grad = True
#%% Overwrite the classifier of the model
# The classifier is the final layer
# Here, we've just 2 classes
model.classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 1)),
    ('Output', nn.Sigmoid())
]))
model.to("mps")
#%% train the model
opt = optim.Adam(model.classifier.parameters())
loss_function = nn.BCELoss()
train_losses = []

model.train()
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    train_loss = 0
    for bat, (img, label) in enumerate(train_loader):
        # Zeroing gradients
        opt.zero_grad()

        # forward pass
        output = model(img.to("mps"))

        # calc losses
        loss = loss_function(output.squeeze(), label.float().to("mps"))

        # Backward
        loss.backward()

        # update weights
        opt.step()

        # update current train loss
        train_loss += loss.item()

    train_losses.append(train_loss)
    print(f"epoch: {epoch}, train_loss: {train_loss}")

#%% show losses over epoch
sns.lineplot(x=range(len(train_losses)), y=train_losses)


# %% Testing
fig = plt.figure(figsize=(10, 10))
class_labels = {0:'cat', 1:'dog'}
X_test, y_test = next(iter(test_loader))
with torch.no_grad():
    y_pred = model(X_test.to("mps"))
    y_pred = y_pred.round()
    y_pred = [p.item() for p in y_pred]

# create subplots
for num, sample in enumerate(X_test):
    if num < (4*6):
        plt.subplot(4,6,num+1)
        plt.title(class_labels[y_pred[num]])
        plt.axis('off')
        sample = sample.cpu().numpy()
        plt.imshow(np.transpose(sample, (1,2,0)))

#%% accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {np.round(acc * 100, 2)} %")



# %%  Explain some decisions with GradCam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

transform = transforms.ToTensor()

im = Image.open('test/cat/2613.jpg')
im = np.float32(im) / 255

# set the target_layers to trainable
#for params in model.parameters():
#   params.requires_grad = True

model.features[-1].weight.requires_grad = True

target_layers = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)
targets = [ClassifierOutputTarget(0)]
#input_tensor = transforms.ToTensor()(im).unsqueeze_(0)
input_tensor = transform(im).unsqueeze(0)
grayscale_cam = cam(input_tensor=input_tensor.to("mps"), targets=targets, aug_smooth=True, eigen_smooth=True)
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(im, grayscale_cam, use_rgb=True)
Image.fromarray(visualization )

# %% save model state dict
torch.save(model.state_dict(), 'model_state_dict.pth')

# %% load a model
#model = models.densenet121(weights='DenseNet121_Weights.DEFAULT')

#%% Freeze all layers
#for params in model.parameters():
#    params.requires_grad = False

#model.classifier = nn.Sequential(OrderedDict([
#    ('fc1', nn.Linear(1024, 1)),
#    ('Output', nn.Sigmoid())
#]))
#model.to("mps")
# model.state_dict()  # randomly initialized
#model.load_state_dict(torch.load('model_state_dict.pth'))

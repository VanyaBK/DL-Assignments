#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# mount the google drive to Colab
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)


# In[ ]:


datasetpath = "/content/drive/My Drive/CS6910_PA1/4"  # e.g., /content/drive/My Drive/CS6910_PA1/1


# In[ ]:


# Imports
import torch, os, os.path as osp
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np


# In[ ]:


# make sure, colab is using GPU. If not, Edit --> Notebook settings --> Set "Hardware accelerator" to GPU
print(torch.cuda.is_available())


# In[ ]:


# API to load the saved dataset file
def get_dataloader_from_pth(path, batch_size=4):
    print('loading {}'.format(path))
    contents = torch.load(path)
    print('data split: {}, classes: {}, {} data points'.format(contents['split'], contents['classes'], len(contents['x'])))

    # create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(contents['x'], contents['y'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)
    
    return dataloader, contents['classes']


# In[ ]:


# file paths
train_pth = osp.join(datasetpath, 'train.pth')
val_pth = osp.join(datasetpath, 'val.pth')
test_pth = osp.join(datasetpath, 'test.pth')


# In[ ]:


# create dataloaders
trainloader, classes = get_dataloader_from_pth(train_pth, batch_size=4)
valloader, _ = get_dataloader_from_pth(val_pth, batch_size=4)
testloader, _ = get_dataloader_from_pth(test_pth, batch_size=4)


# In[ ]:



# Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# <<<<<<<<<<<<<<<<<<<<< EDIT THE MODEL DEFINITION >>>>>>>>>>>>>>>>>>>>>>>>>>
# Try experimenting by changing the following:
# 1. number of feature maps in conv layer
# 2. Number of conv layers
# 3. Kernel size
# etc etc.,


# In[ ]:


#num_epochs = 30         # desired number of training epochs.
#learning_rate = 0.001   

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5)
        #self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,kernel_size=1)
        self.fc1 = nn.Linear(in_features=256, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=33) 
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #print(x.size())
        #x = (F.relu(self.conv4(x)))
        #print(x.size())
        x = F.avg_pool2d(x, kernel_size=x.shape[2:])
        #print(x.size())
        x = x.view(x.shape[0], -1)
        #print(x.size())
        #x = x.view(-1,256*5*5)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

################### DO NOT EDIT THE BELOW CODE!!! #######################

#net = ResNet()
net = Net()

# transfer the model to GPU
if torch.cuda.is_available():
    net = net.cuda()


# In[ ]:


########################################################################
# Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()

learning_rate = 0.0009
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

num_params = np.sum([p.nelement() for p in net.parameters()])
print(num_params, ' parameters')


# In[ ]:


########################################################################
# Train the network
# ^^^^^^^^^^^^^^^^^^^^

def train(epoch, trainloader, optimizer, criterion):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(tqdm(trainloader), 0):
        # get the inputs
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the train images: %d %%' % (
                                    100 * correct / total))
    print('\nepoch %d training loss: %.3f' %
            (epoch + 1, running_loss / (len(trainloader))))
    


# In[ ]:


########################################################################
# Let us look at how the network performs on the test dataset.

def test(testloader, model, set_name):
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()        
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    accuracy = 100 * correct / total
    print('\nAccuracy of the network on the %s images: %d %%' % (set_name, accuracy))
    print('epoch %d test loss: %.3f' %
            (epoch + 1, running_loss / (len(testloader))))
    return accuracy


# In[ ]:



def classwise_test(testloader, model):
########################################################################
# class-wise accuracy
    n_class = len(classes) # number of classes

    class_correct = list(0. for i in range(n_class))
    class_total = list(0. for i in range(n_class))
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()        
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            
    print('\nclass-wise accuracy')
    for i in range(n_class):
        print('Accuracy of %10s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


# In[ ]:


print('\nStart Training')
os.makedirs('./models', exist_ok=True)

num_epochs = 50        # desired number of training epochs.
best_model_epoch = 0
best_accuracy = 0

for epoch in range(num_epochs):  # loop over the dataset multiple times
    print('\nepoch ', epoch + 1)
    train(epoch, trainloader, optimizer, criterion)
    accuracy = test(valloader, net, 'validation')
    # classwise_test(valloader, net)

    # save model checkpoint 
    torch.save(net.state_dict(), './models/model1-'+str(epoch)+'.pth')
    if accuracy > best_accuracy: best_model_epoch = epoch

print('Finished Training')


# In[ ]:


import torch
model = torch.load("models/model1-13.pth") #Best Model from validation accuracy values


# In[ ]:


#PART - B Ques 2.1
#Adding patches to all images in testdata for N=30 and i,j(mid-point of the patch) varying from 15 to 45 
import matplotlib.pyplot as plt

images = []

for I in range(30):
  for J in range(30):
    no_image=0
    image_ij=[]
    for image in testloader:
      temp = image[0][1]
      for k in range(3):
        for i in range(30):
          for j in range(30):
            temp[k][i+I][j+J]=0
      image_ij.append(temp)
      no_image = no_image+1
      if(no_image==10):
        break
    images.append(image_ij)


# In[ ]:


labels=[]
no = 0
for data in testloader:
  image,label = data
  #print(image.size(),label.size())
  labels.append(label[0])
  no = no+1
  if(no==10):
    break


# In[ ]:


true_labels = []
for i in range(30):
  for j in range(30):
    true_labels.append(labels)


# In[ ]:


accuracy = []
for i in range(len(images)):
  X = torch.stack(images[i])
  y=torch.stack(labels)
  dataset = torch.utils.data.TensorDataset(X, y)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                          shuffle=True, num_workers=2)
  accuracy.append(test(dataloader,model,"TEST"))


# In[ ]:


#Plotting the confidence values against the points i,j
points = []
x_points = []
y_points = []
for I in range(30):
  x_points.append(I+15)
  y_points.append(I+15)
x1, y1 = np.meshgrid(x_points, y_points)
accuracy = np.array(accuracy)
accuracy = accuracy.reshape(30,30)
print(accuracy.shape)
import matplotlib.pyplot as plt
plt.contour(x1,y1,accuracy)
plt.show()


# In[ ]:


model1 = Net()
model1.load_state_dict(torch.load("models/model1-13.pth"))


# In[ ]:


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


# In[ ]:


model1.conv3.register_forward_hook(get_activation('conv3'))#Getting ouput of conv3 layer, can be changed to any other layer
output_layer3 = []

ind=0
for data in testloader:
  image,label = data
  output = model1(image)
  output_layer3.append(activation['conv3'])
  


# In[ ]:


import matplotlib.pyplot as plt
import collections
def visualise(filter):
  patch = collections.defaultdict(list)
  H = list(output_layer3[0].size())[2]
  W = list(output_layer3[0].size())[3]
  for i in range(len(output_layer3)):
    for j in range(4):
      for h in range(H):
        for w in range(W):
          patch[output_layer3[i][j][filter][h][w]].append([i,j,h,w])
  max_response = sorted(patch.keys(),reverse=True)
  indices = []
  for i in range(5):
    print(patch[max_response[i]])
    indices.append(patch[max_response[i]][0][0])
  '''
  size, stride and start are calculated from the following receptive area formulae :
  jump = jump(input)*stride
  receptive size = receptive size(input) + (kernel size-1)*jump(input)
  start = start(input) + (kernel-1)/2*jump(input)

  From the above formulae
  For conv layer 1
  size = 5
  stride = 1
  start = 2.5

  For conv layer 2
  size = 19
  stride = 3
  start = 10.5

  For conv layer 3
  size = 61
  stride = 9
  start = 34.5

  '''
  size=61   
  stride=9
  start = 34.5
  img_patch = []
  for ind,data in enumerate(testloader):
    image,label = data
    if(ind in indices):
      for i in range(5):
        if(patch[max_response[i]][0][0]==ind):
          batch_no = patch[max_response[i]][0][1]
          h = patch[max_response[i]][0][2]
          w = patch[max_response[i]][0][3]
          start_x = start + stride*h - (size/2.0)
          end_x = start + stride*h + (size/2.0)
          start_y = start + stride*w - (size/2.0)
          end_y = start + stride*w + (size/2.0)
          print(start_x,start_y,end_x,end_y)
          plt.imshow(image[batch_no][:,int(start_x):int(end_x),int(start_y):int(end_y)].permute(1,2,0))
          plt.show()


# In[ ]:


#Filter analysis for two filters is shown here and the same can be repeated for different layers and different filters
visualise(0) #1st filter 3rd layer 
model1.conv3.weight.data[0]=0 #Switching off 1st filter of 3rd layer
classwise_test(testloader,model1)
model1.load_state_dict(torch.load("models/model1-13.pth"))
visualise(1) #2nd filter 3rd layer
model1.conv3.weight.data[1]=0 #Switching off 2nd filter of 3rd layer
classwise_test(testloader,model1)


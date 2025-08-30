import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


# 28x28 images = 784 input features
# Build a feed-forward network
class FashionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        #dropout layer, randomly zeroing some of the elements
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.log_softmax(self.fc4(x), dim=1))
        
        return x

model = FashionClassifier()

criterion = nn.NLLLoss() #Negative Log Likelihood Loss
optimizer = optim.Adam(model.parameters(), lr=0.003)  #Adam optimizer

train_losses, test_losses = [], []
step = 0

epochs = 30
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        #Training pass
        logps = model(images) #Log probabilities
        loss = criterion(logps, labels) #Calculate the Loss
        optimizer.zero_grad() #Zeroing Gradients
        loss.backward() 
        optimizer.step()
        
        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0
        
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)
                 
                ps = torch.exp(log_ps) #Probabilities
                top_p, top_class = ps.topk(1, dim=1) #Top class prediction
                equals = top_class == labels.view(*top_class.shape) #Check for equality
                accuracy += torch.mean(equals.type(torch.FloatTensor)) #Calculate accuracy
        
        model.train()
        
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "Test Loss: {:.3f}.. ".format(test_losses[-1]),
              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
        
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)

dataiter = iter(testloader)
images, labels = next(dataiter)
img = images[1]


# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(model(img))

# Plot the image and predicted probabilities using matplotlib
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(img.squeeze(), cmap='gray')
plt.title("Input Image")
plt.axis('off')

plt.subplot(1,2,2)
classes = trainset.classes
plt.barh(classes, ps.detach().numpy().squeeze())
plt.title("Class Probabilities")
plt.xlabel("Probability")
plt.tight_layout()
plt.show()
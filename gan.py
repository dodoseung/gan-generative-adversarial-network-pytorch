import torch.nn.functional as F
import torch.nn as nn

# Generator of Ganerative Adversarial Network
class Generator(nn.Module):
    def __init__(self, latent_z=100, init_layer=[7,7,64], conv_trans=[2,2,1,1], conv_filters=[128,64,64,1], conv_kernels=[5,5,5,5], conv_strides=[1,1,1,1], dropout_rate=0.1):
        super(Generator, self).__init__()
        # Initiation
        self.init_layer = init_layer
        self.latent_z = latent_z
        self.conv_trans = conv_trans
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.dropout_rate = dropout_rate
        
        # Append filters for the init layer
        self.num_layers = len(self,conv_filters)
        self.conv_filters.insert(0, self.init_layer[2])
        
        # Fully connected layer
        self.fc = nn.Linear(self.latent_z, self.init_layer[0] * self.init_layer[1] * self.init_layer[2])
        self.batch_norm = nn.BatchNorm2d(self.init_layer[2])
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        
        # Convolution layers
        self.conv = []
        for i in range(self.num_layers):
            layer = nn.Sequential(
                nn.ConvTranspose2d(conv_filters[i], conv_filters[i], kernel_size=self.conv_trans[i], stride=self.conv_trans[i]),
                nn.Conv2d(conv_filters[i], conv_filters[i+1], kernel_size=self.conv_kernels[i], stride=self.conv_strides[i], padding='same'),
                nn.BatchNorm2d(conv_filters[i+1]),
                nn.ReLU())
            self.conv.append(layer)
        self.conv = nn.ModuleList(self.conv)
  
    def forward(self, z):
        # FCN
        x = self.fc(z)
        x = F.relu(self.batch_norm(x))
        
        # Reshape and drop out
        x = x.view(-1, self.init_layer[2], self.init_layer[0], self.init_layer[1])
        x = self.dropout(x)
        
        # CNN
        for _ in range(self.num_layers):
            x = self.conv(x)
        
        return x
    
# Discriminator of Ganerative Adversarial Network
class Discriminator(nn.Module):
    def __init__(self, input_img=[28,28,1], conv_filters=[64,64,128,128], conv_kernels=[5,5,5,5], conv_strides=[2,2,2,1], dropout_rate=0.4):
        super(Discriminator, self).__init__()
        # Initiation
        self.input_img = input_img
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.dropout_rate = dropout_rate
        
        # Append filters for the input image
        self.num_layers = len(self,conv_filters)
        self.conv_filters.insert(0, self.input_img[2])
        
        # Convolution layers
        self.conv = []
        for i in range(self.num_layers):
            layer = nn.Sequential(
                nn.Conv2d(conv_filters[i], conv_filters[i+1], kernel_size=self.conv_kernels[i], stride=self.conv_strides[i], padding='same'),
                nn.BatchNorm2d(conv_filters[i+1]),
                nn.ReLU(),
                nn.Dropout(rate=self.dropout_rate))
            self.conv.append(layer)
        self.conv = nn.ModuleList(self.conv)
        
        # Output layer
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.conv_filters[-1], 1)
        
    def forward(self, x):
        # CNN
        for _ in range(self.num_layers):
            x = self.conv(x)
            
        # Output layer
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        
        return x



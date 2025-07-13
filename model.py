import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

from masks import Mask

class NeuralNetwork(nn.Module):
    def __init__(self, num_dendrites, num_somas, mask_type=None, masking=True):
        super().__init__()
        self.num_inputs = 28*28
        self.image_length = 28
        self.inputs_per_dendrite = 16
        self.num_dendrites = num_dendrites
        self.num_somas = num_somas
        self.num_outputs = 10
        self.masking = masking # if False, no masks are applied --> vANN
        if self.masking == True:
            self.mask_type = mask_type if mask_type!=None else "random"

        # layers of the dANN
        self.input_layer = nn.Linear(self.num_inputs, self.num_dendrites)
        self.dendritic_layer = nn.Linear(self.num_dendrites, self.num_somas)
        self.soma_layer = nn.Linear(self.num_somas, self.num_outputs)

        if self.masking == True:
            # create binary masks
            self.register_buffer('mask_input_dendrites', torch.from_numpy(Mask.create_mask_input_to_dendrites(self.num_inputs, self.num_dendrites, self.mask_type, self.inputs_per_dendrite, self.image_length)).float())
            self.register_buffer('mask_dendrites_soma', torch.from_numpy(Mask.create_mask_dendrites_to_soma(self.num_dendrites, self.num_somas)).float())
            # manipalate the initialized weights by the binary masks
            self.input_layer.weight.data = self.input_layer.weight.data * self.mask_input_dendrites
            self.dendritic_layer.weight.data = self.dendritic_layer.weight.data * self.mask_dendrites_soma

        self.flatten = nn.Flatten()


    def forward(self, x):
        
        x = self.flatten(x)
            
        dendrites = self.input_layer(x)
        dendrites = F.relu(dendrites)

        somas = self.dendritic_layer(dendrites)
        somas = F.relu(somas)

        y = self.soma_layer(somas)
        #y = F.softmax(y, dim=1) # 128x10 batch x output
        
        return y
    
    def apply_masking(self):
        if self.masking == True:
            with torch.no_grad():
                self.input_layer.weight.data = self.input_layer.weight.data * self.mask_input_dendrites
                self.dendritic_layer.weight.data = self.dendritic_layer.weight.data * self.mask_dendrites_soma


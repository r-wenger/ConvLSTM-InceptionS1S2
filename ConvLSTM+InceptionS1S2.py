import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

######################################
# MODELS
######################################

class ConvLSTM2D(nn.Module):
    """
    Convolutional LSTM layer for spatial-temporal data processing.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, padding, batch_first=False):
        super(ConvLSTM2D, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.batch_first = batch_first
        
        # Convolutional layer for LSTM gates
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, 1, padding)
        self.batchnorm = nn.BatchNorm2d(hidden_dim)

    def forward(self, x, state):
        # Define batch-first or sequence-first processing
        if self.batch_first:
            batch_size, seq_len, height, width, channels = x.size()
        else:
            seq_len, batch_size, height, width, channels = x.size()
            
        h, c = state
        output = []

        # Process each time step
        for t in range(seq_len):
            x_t = x[:, t, :, :, :] if self.batch_first else x[t, :, :, :, :]
            x_t = x_t.permute(0, 3, 1, 2)  # Convert NHWC to NCHW
            combined = torch.cat([x_t, h], dim=1)
            gates = self.conv(combined)

            # Split the gates
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)
        
            # Update cell and hidden states
            c = forgetgate * c + ingate * cellgate
            h = outgate * torch.tanh(c)
            h = self.batchnorm(h)
            output.append(h)
        
        # Stack outputs along the time dimension
        output = torch.stack(output, dim=1 if self.batch_first else 0)
        return output, (h, c)

class InceptionModule(nn.Module):
    """
    Simplified Inception module for multi-scale feature extraction.
    """
    def __init__(self, in_channels):
        super(InceptionModule, self).__init__()
        # Path 1: 1x1 followed by 3x3 convolution
        self.tower1_1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.tower1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # Path 2: 1x1 followed by 5x5 convolution
        self.tower2_1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.tower2_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        # Path 3: MaxPooling followed by 1x1 convolution
        self.tower3_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.tower3_2 = nn.Conv2d(in_channels, 32, kernel_size=1)

    def forward(self, x):
        tower1 = F.relu(self.tower1_1(x))
        tower1 = F.relu(self.tower1_2(tower1))
        
        tower2 = F.relu(self.tower2_1(x))
        tower2 = F.relu(self.tower2_2(tower2))
        
        tower3 = self.tower3_1(x)
        tower3 = F.relu(self.tower3_2(tower3))
        
        # Concatenate along the channel dimension
        output = torch.cat([tower1, tower2, tower3], dim=1)
        return output

class FuseLSTM(nn.Module):
    """
    Fusion model combining ConvLSTM, Inception, and U-Net for multimodal data.
    """
    def __init__(self):
        super(FuseLSTM, self).__init__()
        # ConvLSTM blocks for S2 and S1 inputs
        self.convLSTM_s2 = ConvLSTM2D(input_dim=10, hidden_dim=32, kernel_size=3, padding=1, batch_first=True)
        self.convLSTM_s1 = ConvLSTM2D(input_dim=2, hidden_dim=32, kernel_size=3, padding=1, batch_first=True)
        
        # Inception module for stack input
        self.inception = InceptionModule(in_channels=12)
        
        # U-Net for final segmentation
        self.unet = smp.Unet(
            encoder_name="vgg16", 
            encoder_weights=None,
            in_channels=160,  # Input channels to U-Net
            classes=10,  # Number of output classes
            activation="softmax"
        )
        
        # Freeze U-Net encoder
        for param in self.unet.encoder.parameters():
            param.requires_grad = False

    def forward(self, x_s2, x_s1, x_stack):
        # Initialize hidden and cell states for ConvLSTM
        batch_size, _, height, width, _ = x_s2.size()
        h0 = torch.zeros(batch_size, 32, height, width).to(x_s2.device)
        c0 = torch.zeros(batch_size, 32, height, width).to(x_s2.device)
        initial_state = (h0, c0)

        # Process S2 and S1 sequences with ConvLSTM
        x_s2, _ = self.convLSTM_s2(x_s2, initial_state)
        x_s1, _ = self.convLSTM_s1(x_s1, initial_state)

        # Process stack input with Inception module
        x_stack = x_stack.permute(0, 3, 1, 2)  # Convert NHWC to NCHW
        x_stack = self.inception(x_stack)

        # Extract last time step outputs from ConvLSTM
        x_s2 = x_s2[:, -1, ...]
        x_s1 = x_s1[:, -1, ...]

        # Concatenate features from S2, S1, and stack
        x = torch.cat([x_s2, x_s1, x_stack], dim=1)

        # Pass through U-Net for segmentation
        x = self.unet(x)
        return x

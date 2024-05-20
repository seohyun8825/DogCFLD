import torch.nn as nn
from diffusers.models.resnet import ResnetBlock2D, Downsample2D

class PoseEncoder(nn.Module):
    def __init__(self, downscale_factor, pose_channels, in_channels, channels, groups=32):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        unshuffle_output_channels = pose_channels * (downscale_factor ** 2)
        #print(f"Expected conv_in input channels: {unshuffle_output_channels}")  # Debugging line

        # Ensure conv_in input channels match unshuffled output channels
        self.conv_in = nn.Conv2d(unshuffle_output_channels, in_channels, kernel_size=1)

        resnets = []
        downsamplers = []
        for i in range(len(channels)):
            in_channels = in_channels if i == 0 else channels[i - 1]
            out_channels = channels[i]

            # Ensure num_channels is divisible by groups
            if in_channels % groups != 0 or out_channels % groups != 0:
                groups = 1  # fallback to group size of 1 if not divisible

            resnets.append(ResnetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=None,  # no time embed
                groups=groups
            ))
            downsamplers.append(Downsample2D(
                out_channels,
                use_conv=False,
                out_channels=out_channels,
                padding=1,
                name="op"
            ) if i != len(channels) - 1 else nn.Identity())

        self.resnets = nn.ModuleList(resnets)
        self.downsamplers = nn.ModuleList(downsamplers)

    def forward(self, hidden_states):
        #print(f"Input hidden_states shape: {hidden_states.shape}") 
        hidden_states = self.unshuffle(hidden_states)
        #print(f"Unshuffled hidden_states shape: {hidden_states.shape}")  
        hidden_states = self.conv_in(hidden_states)
        #print(f"Convolved hidden_states shape: {hidden_states.shape}")  
        
        features = []
        for resnet, downsampler in zip(self.resnets, self.downsamplers):
            hidden_states = resnet(hidden_states, temb=None)
            features.append(hidden_states)
            hidden_states = downsampler(hidden_states)
            #print(f"Resnet output hidden_states shape: {hidden_states.shape}")  

        return features

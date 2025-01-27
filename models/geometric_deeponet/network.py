import torch
import torch.nn as nn
from einops import rearrange


class torchSine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class LinearMLP(nn.Module):
    def __init__(self, dims, nonlin):
        super(LinearMLP, self).__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation after the last layer
                layers.append(nonlin())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class GeoDeepONet(nn.Module):
    def __init__(self, input_channels_func, input_channels_loc, output_channels, modes, branch_net_layers=None, trunk_net_layers=None):
        super().__init__()
        
        self.input_channels_func = input_channels_func     # [Re, SDF, Mask]
        self.input_channels_loc = input_channels_loc       # [x, y, SDF]
        self.output_channels = output_channels            # [u,v,p,cd,cl]
        self.modes = modes
        
        self.branch_net_layers_stage_1 = [self.input_channels_func] + branch_net_layers + [self.modes]
        self.branch_net_layers_stage_2 = [self.modes] + branch_net_layers + [self.modes*self.output_channels]
        
        self.trunk_net_layers_stage_1 = [self.input_channels_loc] + trunk_net_layers + [self.modes]
        self.trunk_net_layers_stage_2 = [self.modes] + trunk_net_layers + [self.modes*self.output_channels]
        
        
        self.branch_stage_1 = LinearMLP(dims=self.branch_net_layers_stage_1, nonlin=nn.ReLU)
        self.branch_stage_2 = LinearMLP(dims=self.branch_net_layers_stage_2, nonlin=nn.ReLU)
        
        self.trunk_stage_1 = LinearMLP(dims=self.trunk_net_layers_stage_1, nonlin=nn.ReLU)
        self.trunk_stage_2 = LinearMLP(dims=self.trunk_net_layers_stage_2, nonlin=torchSine)
        
        self.b = torch.tensor(0.0, requires_grad=True)

    def forward(self, x1, x2):
        '''
        x1 : input to branch network. 
                [batch_size, num_points, input_channels_func] 
                num_points = (h * w) of domain
                input_channels_func is number of input function values. 
                For our case that is (Re, SDF, Mask), yes SDF will be redundant for this problem.
        x2 : input to trunk network.
                [batch_size, num_points, input_channels_loc]
                num_points = (h * w) of domain
                input_channels_loc is number of input location values.
                For our case that is (x, y, SDF).
        
        We will use the multi_output_strategy of 'split_branch_network' defined in DeepXDE library. 
        https://github.com/lululxvi/deepxde/blob/master/deepxde/nn/deeponet_strategy.py
        
        Our dataloader returns inputs and outputs in the shape of an image [b,h,w,c] where c is input_channels_func/loc.
        So, we need to reshape to get tensor shapes defined above. 
        
        Refer to figure 3 in https://arxiv.org/pdf/2403.14788 for "stage" naming convention to define separate networks
        
        einops notation:
        b - batch size
        h - height of domain
        w - width of domain
        c - channels (number of values, will change for inputs/outputs and different networks)
        m - modes
        p - num_pts (h*w)
        '''
        
        x1_flattened = rearrange(x1, 'b c h w -> b (h w) c')
        x2_flattened = rearrange(x2, 'b c h w -> b (h w) c')
        
        intermediate_branch = self.branch_stage_1(x1_flattened)
        intermediate_trunk = self.trunk_stage_1(x2_flattened)
        
        # element-wise multiplication to merge intermediate representations from branch and trunk networks
        merge_intermediate_reps = torch.einsum('bpm, bpm -> bpm', intermediate_branch, intermediate_trunk)
        # average operation over merged intermediate representations for branch network only
        avg_merged_reps_branch_net = torch.mean(merge_intermediate_reps, dim=1) # [batch, num_pts, modes] -> [batch, avg_modes]
        
        output_branch = self.branch_stage_2(avg_merged_reps_branch_net)
        output_trunk = self.trunk_stage_2(merge_intermediate_reps)
        
        # reshape tensors so branch output is shape [batch, modes, output_channels]
        # and trunk output is shape [batch, num_pts, modes, output_channels]
        output_branch = rearrange(output_branch, 'b (m c) -> b m c', m=self.modes, c=self.output_channels)
        output_trunk = rearrange(output_trunk, 'b p (m c) -> b p m c', p=x1_flattened.shape[1], m=self.modes, c=self.output_channels)
        
        # conduct final dotproduct over outputs from each network to get predicted solution
        output_solution = torch.einsum('bmc, bpmc -> bpc', output_branch, output_trunk)
        output_solution = output_solution + self.b
        
        # reshape so output is of shape [batch, h, w, c] for loss function in pl.lightning
        output_solution = rearrange(output_solution, 'b (h w) c -> b c h w', h=x1.shape[-1], w=x1.shape[-1], c=self.output_channels)

        return output_solution
    
    

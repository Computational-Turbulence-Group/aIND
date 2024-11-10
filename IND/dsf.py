import numpy as np

import torch
from torch import nn


################################################################################
# The following functions have been copied from:
#   https://github.com/CW-Huang/torchkit
delta = 1e-6
softplus_ = nn.Softplus()
softplus = lambda x: softplus_(x) + delta
sigmoid_ = nn.Sigmoid()
sigmoid = lambda x: sigmoid_(x) * (1-delta) + 0.5 * delta
logsigmoid = lambda x: -softplus(-x)
log = lambda x: torch.log(x*1e2)-np.log(1e2)
logit = lambda x: log(x) - log(1-x)
def softmax(x, dim=-1):
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    out = e_x / e_x.sum(dim=dim, keepdim=True)
    return out


class SigmoidFlow(nn.Module):
    '''DSF neural network from Huang et al (2018)

    This function is a simplified version of the one in the repository:
        https://github.com/CW-Huang/torchkit

    Huang, C-W, et al (2018) Neural autoregressive flows
    https://arxiv.org/abs/1804.00779
    '''
    def __init__(self, num_neurons: int ):
        '''Init a SigmoidFlow layer from Huang et al. (2018)

        Parameters:
            num_neurons: int
                Number of neurons in the layer

        '''
        super().__init__()

        # Parameters a, b and w without constraints. nn.Parameter
        # has requries_grad = True by default
        self.a0 = nn.Parameter( torch.randn( num_neurons, 1 ) )
        self.b0 = nn.Parameter( torch.randn( num_neurons, 1 ) )
        self.w0 = nn.Parameter( torch.randn( 1, num_neurons ) )

        # constraints for the weights as activation functions
        self.act_a = lambda x: softplus(x)       # force positive
        self.act_b = lambda x: x                 # do nothing
        self.act_w = lambda x: softmax(x,dim=-1) # force > 0 and sum = 1


    def forward(self, x, mollify: float = 0.0, delta: float = delta ):
        '''Implement equation 8 from Huang et al. (2018)

            y = logit ( w * sigmoid ( a * x + b ) )

        Parameters:
            x: torch.tensor
                Input array [ 1 , N ] with N = number of samples

            mollify: float
                Parameter to smooth a and b. It was in the original repo, but set
                to 0 (no smoothing)

            delta: float
                Parameter to make x > 0 (I think) set to default value from repo

        Returns:
            y: torch.tensor
                Output array [ 1, N ]

        '''

        # Apply constraints to the weights
        a_ = self.act_a( self.a0 )
        b_ = self.act_b( self.b0 )
        w  = self.act_w( self.w0 )

        # This was included in the original git repo. Mollify is 0, so it shouldn't matter
        a = a_ * (1-mollify) + 1.0 * mollify
        b = b_ * (1-mollify) + 0.0 * mollify

        # Equation 8 in Huang 2018 et al
        sigm = torch.sigmoid( a * x + b )
        #x_pre = torch.sum( w*sigm ) # original (the latter should be equivalent)
        x_pre = torch.matmul( w, sigm )

        x_pre_clipped = x_pre * (1-delta) + delta * 0.5

        # Apply the inverse sigmoid
        return logit( x_pre_clipped )


    def get_weights(self):
        '''return the weights after applying the constraints
            a, w, b = get_weights()
        '''
        return self.act_a( self.a0 ), self.act_w( self.w0 ), self.act_b ( self.b0 )


class DDSF_layer(nn.Module):
    '''DDSF neural network from Huang et al (2018)

    This function is a simplified version of the one in the repository:
        https://github.com/CW-Huang/torchkit

    Huang, C-W, et al (2018) Neural autoregressive flows
    https://arxiv.org/abs/1804.00779
    '''

    def __init__(self, n_input, n_output, num_neurons = -1):
        super().__init__()
        '''Init a Deep Dense Sigmoid Flow (DDSF) layer from Huang et al. (2018)

        Parameters:
            n_input: int
                Number of inputs from the previous layer

            n_output: int
                Number of outputs

            num_neurons: int
                Number of neurons in the layer. Note that, unless this is the last
                layer, num_neurons = n_output. In the last layer n_output = 1 and
                num_neurons = n_input
        '''
        if num_neurons == -1: num_neurons = n_output

        # Parameters a, b and w without constraints. nn.Parameter
        # has requries_grad = True by default
        self.a0 = nn.Parameter( torch.randn( num_neurons, n_input ) )
        self.b0 = nn.Parameter( torch.randn( num_neurons, n_input ) )
        self.w0 = nn.Parameter( torch.randn( n_output, num_neurons ) )
        self.u0 = nn.Parameter( torch.randn( num_neurons, n_input ) )

        # constraints for the weights as activation functions
        self.act_a = lambda x: softplus(x)       # force positive
        self.act_b = lambda x: x                 # do nothing
        self.act_w = lambda x: softmax(x,dim=-1) # force > 0 and sum = 1
        self.act_u = lambda x: softmax(x,dim= 0) # force > 0 and sum = 1


    def forward(self, x, mollify=0.0, delta=delta):
        '''Implement equation 9 from Huang et al. (2018)

            y = logit ( w * sigmoid ( a * (u * x) + b ) )

        Parameters:
            x: torch.tensor
                Input array [ n_input , N ] with N = number of samples

            mollify: float
                Parameter to smooth a and b. It was in the original repo, but set
                to 0 (no smoothing)

            delta: float
                Parameter to make x > 0 (I think) set to default value from repo

        Returns:
            y: torch.tensor
                Output array [ n_output, N ]

        '''
        # Apply constraints to the weights
        a_ = self.act_a( self.a0 )
        b_ = self.act_b( self.b0 )
        w  = self.act_w( self.w0 )
        u  = self.act_u( self.u0 )

        # This was included in the original git repo. No idea why (Stability?)
        # mollify is 0, so it shouldn't matter
        a = a_ * (1-mollify) + 1.0 * mollify
        b = b_ * (1-mollify) + 0.0 * mollify

        # Equation 8 in Huang 2018 et al
        x_pre = torch.matmul( u, x )

        x_pre = (a[:,:,None] * x_pre[None,:,:] + b[:,:,None]).sum(1)
        sigm = torch.sigmoid( x_pre )
        x_pre = torch.matmul( w, sigm )

        x_pre_clipped = x_pre * (1-delta) + delta * 0.5

        # Apply the inverse sigmoid
        return logit( x_pre_clipped )


    def get_weights(self):
        '''return the weights after applying the constraints
            a, w, b, u = get_weights()
        '''
        return self.act_a( self.a0 ), self.act_w( self.w0 ), \
                self.act_b ( self.b0 ), self.act_u ( self.u0 )


class DSF(nn.Module):

    def __init__( self, num_layers: int, num_neurons: tuple | int ):
        '''DSF layers stack sequentially as in Fig 4b of Huang et al. (2018)

        Parameters:
            num_layers: int
                Number of layers to stack

            num_neurons: tuple | int
                Number of neurons per layers. If int, all layers have the same
                number of neurons. If list, size must be ( num_layers, ) -> the
                i-th entry is the number of neurons in the i-th layer

        '''
        super().__init__()

        self.layers = nn.ModuleList()

        # DSF can only handle increasingly monotonic function, so
        # we premultiply the input by a variable bounded between
        # -1 and 1 (tanh)
        self.s0 = nn.Parameter( torch.randn( 1 ) ) # sign parameter
        self.act_s = lambda x: torch.tanh(x)     # force -1 < x < 1

        # If num_neurons is not a list, all layers have the same humber of neurons:
        if type(num_neurons) == int:
            num_neurons = (num_neurons,) * num_layers
        elif type(num_neurons) == tuple:
            assert( len( num_neurons ) == num_layers )
        else:
            raise ValueError('num_neurons must be an integer or a tuple with' +
                             f' {num_layers} elements')

        self.layers = nn.ModuleList()
        for i in range( num_layers ):
            self.layers.append( SigmoidFlow( num_neurons[i] ) )


    def forward(self, x):
        '''Compute output of stacked layers

            y = layer[-1]( layer[-2]( ... ( layer[0]( x ) ) ) )

        Parameters:
            x:  torch.tensor
                Input array [ 1, N ]. Where N is the number of samples

        Returns:
            y: torch.tensor
                Output array [ 1, N ].

        '''

        s  = self.act_s( self.s0 )
        # Premultiply input by sign
        x = s * x

        for l in self.layers:
            x = l (x)

        return x



class DDSF(nn.Module):

    def __init__(self, num_layers: int = 5, num_neurons: int = 8):
        '''DDSF layers stack sequentially as in Fig 4b of Huang et al. (2018)

        Parameters:
            num_layers: int
                Number of layers to stack

            num_neurons: int
                Number of neurons per layers (all layers have the same number of
                                              neurons)

        '''
        super().__init__()

        # DSF can only handle increasingly monotonic function, so
        # we premultiply the input by a variable bounded between
        # -1 and 1 (tanh)
        self.s0 = nn.Parameter( torch.randn( 1 ) ) # sign parameter
        self.act_s = lambda x: torch.tanh(x)     # force -1 < x < 1

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append( DDSF_layer( 1, num_neurons ) )
        for _ in range( 1, num_layers ):
            self.layers.append( DDSF_layer( num_neurons, num_neurons ) )

        # Last layer has 1 output.
        self.layers.append( DDSF_layer( num_neurons, 1, num_neurons = num_neurons ) )


    def forward( self, x ):
        '''Compute output of stacked layers

            y = layer[-1]( layer[-2]( ... ( layer[0]( x ) ) ) )

        Parameters:
            x:  torch.tensor
                Input array [ 1, N ]. Where N is the number of samples

        Returns:
            y: torch.tensor
                Output array [ 1, N ].

        '''

        s  = self.act_s( self.s0 )
        # Premultiply input by sign
        x = s * x

        for l in self.layers:
            x = l(x)

        return x

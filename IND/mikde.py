import torch
from torch import nn

################################################################################
# The following class have been copied and modified from
#   https://github.com/connorlee77/pytorch-mutual-information.git
class MutualInformation(nn.Module):
    '''class to compute the mutual information between torch arrays

        MutualInformation( sigma, num_bins, lim, normalize, device )

    The data is smoothed using a gaussian kernel:
        K( x ) = exp( -0.5 * (x - xi)**2 / sigma ** 2 )
    where xi are coordinates specified by bins

    Parameters ( optional, default values in parentheses )

        num_bins    [int] (*256)
            Number of bins to split the data (xi points)

        sigma       [float | str] (*0.1)
            bandwidth -> value, "scott" or "silverman"

        lim         [int]   (*4)
            limits of xi ( -lim, lim ). Note we assume signals have 0 mean

        normalize   [bool]  (*true)
            If true, MI is normalized with ( Hx1 + Hx2 ) / 2

        device   [str]  (*'cpu')
            where the computations are performed

    Adapted from:
    https://github.com/connorlee77/pytorch-mutual-information.git
    '''

    epsilon = 1e-10 # mollify value for pdfs

    def __init__(self, num_bins: int, sigma: float | str =0.1, lim: float = 4.,
                 normalize: bool= True, device: str | torch.device ='cpu'):
        super(MutualInformation, self).__init__()
        '''Init mutual information object

            MutualInformation( sigma, num_bins, lim, normalize, device )

        The data is smoothed using a gaussian kernel:
            K( x ) = exp( -0.5 * (x - xi)**2 / sigma ** 2 )
        where xi are coordinates specified by bins

        Parameters ( optional, default values in parentheses )

            num_bins    [int]
                Number of bins to split the data (xi points)

            sigma       [float | str] (*0.1)
                bandwidth -> value, "scott" or "silverman"

            lim         [int]   (*4)
                limits of xi ( -lim, lim ). Note we assume signals have 0 mean

            normalize   [bool]  (*true)
                If true, MI is normalized with ( Hx1 + Hx2 ) / 2

            device   [str]  (*'cpu')
                where the computations are performed

        '''
        self.sigma_method = None
        if isinstance(sigma, str ):
            self.sigma_method = sigma   # bandwidth
        else:
            self.sigma = torch.tensor( [sigma] )[0].to(device) # make it a tensor of dim 0 in cuda

        self.num_bins = num_bins        # number of bins

        self.normalize = normalize
        self.device = device

        # Init xi coordinates
        self.bins = nn.Parameter(torch.linspace(-lim, lim, num_bins,
                                                device=device).float(),
                                 requires_grad=False)


    def _update_sigma( self, n_: int, method:str ):

        d = 1   # number of dimensions
        n = float( n_ )

        sigma = -100000.0
        if method == 'scott':
            sigma = n ** ( -1./( d + 4. ) )
        elif method == 'silverman':
            sigma = (n * (d + 2.) / 4.)**(-1. / (d + 4.))
        else:
            print( 'value of sigma not update.',
                  'method must be "scott" or "silverman"' )

        self.sigma = torch.tensor( [sigma] )[0].to(self.device) # make it a tensor of dim 0 in cuda


    def marginalPdf(self, values):
        '''Compute marginal pdfs (i.e., px and py) using gaussian kde

            pdf, kernel_values = self.marginalPdf( values )

        Parameters
            values  [ torch.Tensor ]
                values of the variable. shape [ N x 1 ]

        Returns
            pfd     [ torch.Tensor ]
                value of the pdf evaluated at xi points [ nbins x 1 ]

            kernel_values   [ torch.Tensor ]
                average value of the gaussian kde at xi points [ nbins x 1 ]
        '''

        residuals = values - self.bins.unsqueeze(0) # (x - xi)
        kernel_values = torch.exp(-0.5*(residuals / self.sigma).pow(2))
        # each row = value of the gaussian filter at all xi for a single point

        # sum the contribution of all points to a single xi
        pdf = torch.mean(kernel_values, dim=0)
        normalization = torch.sum(pdf) + self.epsilon   # ensure >0 values
        pdf = pdf / normalization

        # NOTE: gaussian kde would require to divide the kernel_values
        # by the bandwidth and sqrt( 2 * pi ). But since we are gonna
        # compute the entropies, these constant does not affect the
        # result
        return pdf, kernel_values


    def jointPdf(self, kernel_values1, kernel_values2):
        '''Compute the joint pdf ( pxy ) uning gaussian kde

            pdf = self.jointPdf( kernel_values1, kernel_values2 )

        The kde assumes local independent gaussian distribution at (xi,xi)

        Parameters
            kernel_values1  [ torch.Tensor ]
                average value of the gaussian kde at xi points for X [ nbins x 1 ]

            kernel_values2  [ torch.Tensor ]
                average value of the gaussian kde at xi points for Y [ nbins x 1 ]

        Returns
            pfd     [ torch.Tensor ]
                value of the pdf evaluated at (xi,xi) points [ nbins x nbins ]
        '''

        # p(x,y) = p(x) * p(y) -> locally independent
        joint_kernel_values = torch.matmul(kernel_values1.T, kernel_values2)
        normalization = joint_kernel_values.sum() + self.epsilon
        pdf = joint_kernel_values / normalization

        return pdf


    def getMutualInformation(self, input1: torch.Tensor, input2: torch.Tensor)->torch.Tensor:
        '''Return the MI in bits between 2 variables using a gaussian kde

            mi = self.getMutualInformation( input1, input2 ) 

        Parameters:
            input1      [ torch.Tensor ]
                input variable 1 [ 1 x N ]

            input2      [ torch.Tensor ]
                input variable 2 [ 1 x N ]

        Returns
            mi      [ torch.Tensor ]
                mutual information between input1 and input2 in nats (float [])
        '''

        assert((input1.shape == input2.shape))

        x1 = input1.view( input1.shape[::-1] )
        x2 = input2.view( input2.shape[::-1] )

        if self.sigma_method is not None:
            self._update_sigma( x1.shape[0], self.sigma_method )

        pdf_x1, kernel_values1 = self.marginalPdf( x1 )
        pdf_x2, kernel_values2 = self.marginalPdf( x2 )
        pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

        H_x1 = -torch.sum(pdf_x1*torch.log2(pdf_x1 + self.epsilon) )
        H_x2 = -torch.sum(pdf_x2*torch.log2(pdf_x2 + self.epsilon) )
        H_x1x2 = -torch.sum(pdf_x1x2*torch.log2(pdf_x1x2 + self.epsilon) )
        mutual_information = H_x1 + H_x2 - H_x1x2

        if self.normalize: mutual_information = 2*mutual_information/(H_x1+H_x2)

        return mutual_information


    def forward(self, input1, input2):
        '''Return the MI in bits between 2 variables using a gaussian kde

            mi = self.getMutualInformation( input1, input2 ) 

        Parameters:
            input1      [ torch.Tensor ]
                input variable 1 [ 1 x N ]

            input2      [ torch.Tensor ]
                input variable 2 [ 1 x N ]

        Returns
            mi      [ torch.Tensor ]
                mutual information between input1 and input2 in nats (float [])
        '''
        return self.getMutualInformation(input1, input2)



class ConditionalMutualInformation( MutualInformation ):

    def jointPdf3(self, kernel_values1, kernel_values2, kernel_values3):

        nn = self.num_bins
        # p(x,y,z) = p(x) * p(y) * p(z) -> locally independent
        joint_kernel_values = torch.zeros( (nn, nn, nn) )

        #for i in range(nn):
        #    for j in range(nn):
        #        for k in range(nn):
        #            joint_kernel_values[i,j,k] = (kernel_values1[:,i] * \
        #                    kernel_values2[:,j] * kernel_values3[:,k]).sum()
        # Same result but faster
        for i in range(nn):
            joint_kernel_values[i] = ( kernel_values1[:,[i],None] * \
                   (kernel_values2.view(-1,nn,1) * kernel_values3.view(-1,1,nn)) ).sum(0)

        normalization = joint_kernel_values.sum() + self.epsilon
        pdf = joint_kernel_values / normalization

        return pdf

    def getConditionalMutualInformation(self,
                                        input1: torch.Tensor,
                                        input2: torch.Tensor,
                                        input3: torch.Tensor) ->torch.Tensor:
        '''Return the CMI in bits between 2 variables and cond var using a gaussian kde

            mi = self.getConditionalMutualInformation( input1, input2, input3 ) 

        Parameters:
            input1      [ torch.Tensor ]
                input variable 1 [ 1 x N ]

            input2      [ torch.Tensor ]
                input variable 2 [ 1 x N ]

            input3      [ torch.Tensor ]
                input variable 3 [ 1 x N ]

        Returns
            mi      [ torch.Tensor ]
                cond mutual information between input1 and input2 condition on
                input3 in bits (float [])
        '''

        assert((input1.shape == input2.shape))

        x1 = input1.view( input1.shape[::-1] )
        x2 = input2.view( input2.shape[::-1] )
        x3 = input2.view( input3.shape[::-1] )

        if self.sigma_method is not None:
            self._update_sigma( x1.shape[0], self.sigma_method )

        pdf_x1, kernel_values1 = self.marginalPdf( x1 )
        pdf_x2, kernel_values2 = self.marginalPdf( x2 )
        pdf_x3, kernel_values3 = self.marginalPdf( x3 )

        pdf_x1x3 = self.jointPdf(kernel_values1, kernel_values3)
        pdf_x2x3 = self.jointPdf(kernel_values2, kernel_values3)

        pdf_x1x2x3 = self.jointPdf3(kernel_values1, kernel_values2, kernel_values3)

        H_x1     = -torch.sum(pdf_x1    *torch.log2(pdf_x1     + self.epsilon) )
        H_x2     = -torch.sum(pdf_x2    *torch.log2(pdf_x2     + self.epsilon) )
        H_x3     = -torch.sum(pdf_x3    *torch.log2(pdf_x3     + self.epsilon) )
        H_x1x3   = -torch.sum(pdf_x1x3  *torch.log2(pdf_x1x3   + self.epsilon) )
        H_x2x3   = -torch.sum(pdf_x2x3  *torch.log2(pdf_x2x3   + self.epsilon) )
        H_x1x2x3 = -torch.sum(pdf_x1x2x3*torch.log2(pdf_x1x2x3 + self.epsilon) )

        cond_mutual_information = H_x1x3 + H_x2x3 - H_x1x2x3 - H_x3

        if self.normalize: cond_mutual_information=2*cond_mutual_information/(H_x1+H_x2)

        return cond_mutual_information


    def forward(self, input1, input2, input3):
        '''Return the CMI in bits between 2 variables using a gaussian kde

            mi = self( input1, input2, input3 ) 

        Parameters:
            input1      [ torch.Tensor ]
                input variable 1 [ 1 x N ]

            input2      [ torch.Tensor ]
                input variable 2 [ 1 x N ]

            input3      [ torch.Tensor ]
                conditioned variable 3 [ 1 x N ]

        Returns
            mi      [ torch.Tensor ]
                mutual information between input1 and input2 conditioned on
                input3 in bits (float [])
        '''
        return self.getConditionalMutualInformation(input1, input2, input3)


if __name__ == "__main__":

    s12, s13, s23 = .2, .3, .4
    S = torch.Tensor([[1. , s12, s13],
                      [s12, 1. , s23],
                      [s13, s23, 1. ]])

    ndist = torch.distributions.MultivariateNormal( torch.zeros(3), S )


    X = ndist.sample( (int(1e5),) )


    miobj  = MutualInformation( num_bins= 20, sigma= 'scott', normalize= False )
    cmiobj = ConditionalMutualInformation( num_bins= 20, sigma= 'scott', 
                                           normalize= False )

    # Check low order functions
    miobj._update_sigma(  X.shape[0], miobj.sigma_method )
    cmiobj._update_sigma( X.shape[0], cmiobj.sigma_method )

    px0,  k0  =  miobj.marginalPdf( X[:,[0]] )
    px0c, k0c = cmiobj.marginalPdf( X[:,[0]] )
    px1c, k1c = cmiobj.marginalPdf( X[:,[1]] )
    px2c, k2c = cmiobj.marginalPdf( X[:,[2]] )

    #px0x1x2c = cmiobj.jointPdf3( k0c, k1c, k2c )
    
    
    #mi = miobj( X[:,[0]].view(1,-1), X[:,[1]].view(1,-1) )

    #mi_a = - .5 * torch.log2( torch.tensor([1 - s12**2]) )

    #cmi = cmiobj( X[:,[0]].view(1,-1), X[:,[1]].view(1,-1), X[:,[2]].view(1,-1) )

    #enum = torch.exp( torch.tensor( [1] ) )
    #epi2 = torch.pi * enum * 2
    #ss = torch.tensor( [s12, s13, s23] )
    #h_0_2 = .5 * torch.log2(epi2*( 1 - ss[1]**2 ) )
    #h_0_12 = .5 * torch.log2(epi2*(1+ 2*ss.sum()-(ss**2).sum()/(( 1 - ss[0]**2 ))))

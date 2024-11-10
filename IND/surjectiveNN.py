import torch

from torch import nn
from torch.optim import lr_scheduler

from . import dsf, mikde
from .abstractNN import generalNN, dev2np


################################################################################
# Custom loss functions
################################################################################
class surjMSEloss( nn.Module ):
    '''MSE loss for several functions, p_i:

            loss = sum_i |u - p_i|^2 / |u|^2
    '''
    def __init__(self):
        '''MSE loss for several functions, p_i:

                loss = sum_i |u - p_i|^2 / |u|^2

        Parameters
            unorm:  float 
                equal to |u|^2
        '''
        super( surjMSEloss, self).__init__()


    def forward( self, p: list, u: torch.Tensor ):
        '''Forward pass of the loss function

        Parameters:
            p:  list
                each element is a torch.Tensor of size (1,N) containing
                the output of a NN

            u:  torch.Tensor
                source variable of size (1,N)
        '''
        loss, nsamples = 0.0, 0.0
        for i in range(len(p)):
            loss += ( ( u - p[i] )**2 ).sum() 
            nsamples += u.numel()

        return loss/nsamples


class surjMI_Loss(nn.Module):
    '''Compute the loss function
            f1 * | var - cau |**2 + f2 * MI( cau, res )
    '''

    def __init__(self, miobj: mikde.MutualInformation, 
                 f_mse: float | torch.Tensor,
                 f_mi: float | torch.Tensor ): 
        super(surjMI_Loss, self).__init__()
        '''Compute the loss function
            | var - cau |**2 / f_mse + f_mi * MI( cau, res )

            loss_function = MI_Loss( miobj, f_mse, f_mi )

        Parameters
            miobj:  MutualInformation
                object to compute MI

            f_mse:  float   
                weight factor for MSE

            f_mi:   float
                weight factor for MI
        '''
        self.mi = miobj
        self.loss_fac = f_mse
        self.MI_fac   = f_mi

        self.MSE = surjMSEloss( )


    def forward(self, y_pred, y_, y_pred_s, x_):
        '''Compute loss function

            res = obj( y_pred, y_, x_ )
        
        Parameters
            y_pred:  torch.Tensor 
                prediction of NN (causal contribution)

            y_:  torch.Tensor
                target of NN ( variable to reconstruct )

            y_pred_s:  torch.Tensor 
                added prediction of NN (causal contribution)

            x_:  torch.Tensor 
                variable to compute MI(y_pred,x_) ( residual )
        '''
        loss1 = self.MSE( y_pred, y_ )
        
        #loss2 = 0.0
        #for i in range( len(y_pred) ):
        #    loss2 += self.mi( y_pred[i] , x_[i] )
        loss2 = self.mi( (y_pred_s-y_pred_s.mean()) / y_pred_s.std(), 
                         (x_ - x_.mean()) / x_.std() )

        return loss1*self.loss_fac + loss2.abs()*self.MI_fac, loss1, loss2


################################################################################
# Core NN function
################################################################################
class _surjNNfun( nn.Module ):
    '''Model that uses different bijective NN in different intervals.
    The intervals are also parameters to be optimized
    '''

    def __init__(self, N_b: int, num_layers: int, num_neurons: int, 
                 k: torch.Tensor ):
        '''Init surjective NN model

        The model is composed by N_b bijective functions,
        p_i, that are only effective in a given interval [ xd[i-1], x[i] ]. Outside this
        interval -> p_i[x] = 0.0 for all x not in [ xd[i-1], x[i] ]. The intervals are
        also parameters to be optimized. To force p_i[x] = 0.0, but allowing 
        the optmization of xd, we use the logistic function:
            p_i[x] = p_i[x] * logit( k * ( x - xd[i-1] ) ) * logit( k * ( xd[i] - x ) )

            obj = _surjNNfun( N_b, num_layers, num_neurons, k )

        Parameters:
            N_b:    int
                number of bijective functions

            num_layers: int
                number of layers for each NN

            num_neurons: int 
                number of neurons per layer. 

            k:  float
                logistic growth
        '''
        super().__init__()

        self.Nb = N_b   # number of bijective functions

        # Store NN in list
        self.NNs = nn.ModuleList( [dsf.DSF( num_layers, num_neurons ) 
                                   for _ in range(N_b)] )
        self.xd     = nn.Parameter( torch.randn(N_b-1) )    # init inflection points

        self.register_buffer( 'slope' , k )
        self.register_buffer( 'cutoff', torch.Tensor([.99]) )


    def forward(self, x: torch.Tensor, u: torch.Tensor ):
        '''compute the output of the addition of the NN

        Parameters:
            x: torch.tensor
                Input array ( 1, N ) 

            u: torch.tensor
                Input array ( 1, N ) 

        Returns:
            y: List [torch.tensor]
                List with Nb torch.tensors of size ( 1, N )
        '''
        
        myfac = torch.log( self.cutoff / (1 - self.cutoff ) ) / self.slope
        xd_s, _ = self.xd.sort()

        y = [ self.NNs[i]( x ) for i in range(self.Nb) ]

        xup = xd_s[0] + myfac
        y[0] *= torch.sigmoid( self.slope * ( xup - u ) )
        
        for i in range(1, self.Nb-1):
            xlo = xd_s[i-1] - myfac
            xup = xd_s[i  ] + myfac

            y[i] *= torch.sigmoid( self.slope * ( u - xlo ) )
            y[i] *= torch.sigmoid( self.slope * ( xup - u   ) )

        xlo = xd_s[-1] - myfac

        y[-1] *= torch.sigmoid( self.slope * ( u - xlo ) )
    
        return y



class surjectiveNN( generalNN ):
    

    def __init__(self, N_b: int, num_layers: int, num_neurons: int,
                 run_gpu: bool = False, gpu_id: int=0, slope: float= 800.,
                 loadfrommodel: None | str= None):
        '''Initiliaze the bijective NN object

        Parameters:
            N_b:    int
                number of bijective functions

            num_layers: int
                number of layers for each NN

            num_neurons: int 
                number of neurons per layer. 
            
            run_gpu:    bool
                if True, model is run in gpu. Note that, if no gpus are available,
                the model will still run in cpu

            gpu_id:  int
                id of the gpu to run if cuda.is_available == True

            slope:  float
                logistic growth at the inflection points 

            loadfrommodel:       None or str
                If a string, overwrite the model dimension and init a model with
                sizes of 'loadfrommodel' (only for DSF)
        '''
        generalNN.__init__( self, run_gpu, gpu_id ) 
        
        k = torch.Tensor( [slope] ).to( self.device )

        if loadfrommodel is not None:
            num_layers, num_neurons = self._get_DSF_modelsize( loadfrommodel )
            print( '   Ignoring inputs and initializing a DSF ->  '\
                  f'{num_layers} layers x {num_neurons} neurons/layer' )

        self.model = _surjNNfun( N_b, num_layers, num_neurons, k )


    def initTrainParams(self, **optparams ):
        '''initialize the variables and the optimizer

        Parameters
            **optparams: dic
                Parameters for Adam optimizer
        '''
        self.model.to( self.device )
        self.optimizer = torch.optim.Adam(self.model.parameters(), **optparams)


    def initMIParams( self, num_bins: int= 50, f_mi = .5, f_mse = 1. ):
        miobj, f_mi_n = generalNN._initMIParams( self, num_bins, f_mi = f_mi )

        # Overwrite citerion
        self.criterion = surjMI_Loss( miobj, f_mse, f_mi_n )


    def optimize( self, epochs: int = 10000, patience: int = 10, 
                 lr_scheduler_dic={} ):
        '''train the NN

        Parameters
            epochs: int
                Number of training steps

            tol: int
                Early stop if no improvment is done in the validation set

        '''
        # Tha size of the batches is equal to the total length. We load the
        # variables here, because it is waaaay cheaper.
        q_ , u_  = next(iter(self._tr_l))
        qv_, uv_ = next(iter(self._va_l))

        #               1900  0.52446550   0.52377826  0.06872625 ['0.00998']
        print( '\n      step        loss          MSE          MI         xd' )
        #it, loss0, tol_min = 0, 1e15, False # init step and loss
        it, min_valid_loss, Counter_val = 0, 1000, 0

        scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', 
                                                   verbose=True, 
                                                   **lr_scheduler_dic)

        while ( it < epochs ):

            # Forward pass: Compute predicted y by passing x to the model
            u_pred = self.model( q_, u_ )

            # Zero gradients
            self.optimizer.zero_grad()

            # Compute and print loss
            u_pred_sum = torch.clone( u_pred[0] )
            for i in range(1,len(u_pred)): u_pred_sum += u_pred[i]
            res_ = u_ - u_pred_sum

            loss,_,_ = self.criterion( u_pred, u_, u_pred_sum, res_ )

            # perform a backward pass, and update the weights.
            loss.backward()
            self.optimizer.step()

            # Check if NN has improved
            if it % 100 == 0:
                self.model.eval()     # Optional when not using Model Specific layer

                u_pred = self.model( qv_, uv_ )

                # Compute and print loss
                u_pred_sum = torch.clone( u_pred[0] )
                for i in range(1,len(u_pred)): u_pred_sum += u_pred[i]
                res_ = uv_ - u_pred_sum
                val_loss, e_loss, mi_loss = self.criterion( u_pred, uv_,
                                                           u_pred_sum, res_ )

                # Update the learning rate based on validation loss
                scheduler.step(val_loss)

                if min_valid_loss > val_loss.item():
                    print( f'{it:10g}{val_loss.item():12.8f}',
                           f'{e_loss.item():12.8f}{mi_loss.item():12.8f}',
                           [ f'{x.item():2.5f}' for x in self.model.xd.sort()[0]])  

                    min_valid_loss = val_loss.item()
                else:
                    Counter_val += 1
                    print( f'{it:10g}{val_loss.item():12.8f}',
                           f'{e_loss.item():12.8f}{mi_loss.item():12.8f}',
                           f'( >{min_valid_loss:12.8f})',
                           f' Patience: {Counter_val}/{patience}' )
                    if Counter_val> patience: it = epochs+1

            it += 1

        # Print the result from the optimization
        u_pred = self.model( qv_, uv_ )
        u_pred_sum = torch.clone( u_pred[0] )
        for i in range(1,len(u_pred)): u_pred_sum += u_pred[i]
        res_ = uv_ - u_pred_sum
        loss,_,_ = self.criterion( u_pred, uv_, u_pred_sum, res_ )
        if Counter_val > patience:
            print(f'Optimization finished: Paticence ({patience}) achieved')
        else:
            print(f'Optimization finished: Maximum # steps ({epochs}) reached')

        print('Final error: ', loss.item())
            

    
    def get_cau_perNN( self ):
        '''return the causal variable split in each Nb region
        
            phi, inds = self.get_phi_perNN( )

        Returns
            funcs:  tuple( Nb-1 )
                each entry is a torch.Tensor with the values of phi at
                each timpe step. The entry is nan if is not defined for
                the tuple (q,u)

            inds:   torch.tensor(Nt)
                each value indicates which NN phi belongs to, at each
                time step
        '''

        # Get the validation source and target
        qv_, uv_ = next(iter(self._va_l))

        xd_s, _ = self.model.xd.sort()
        y = [ self.model.NNs[i]( qv_ ) for i in range( self.model.Nb ) ]

        y[0][ y[0] > xd_s[0] ] = torch.nan
        
        for i in range(1, self.model.Nb-1):
            y[i][ (y[i] < xd_s[i-1]) | (y[i] > xd_s[i])] = torch.nan

        y[-1][ y[-1] < xd_s[-1] ] = torch.nan
        
        y = torch.vstack( y ) # make 2d matrix from list

        # get the indices where the difference between u and cau are minimum.
        # For that, replace nan (non valid solutions, with crazy big values)
        inds = torch.argmin( torch.nan_to_num( (y - uv_)**2, nan=1e10 ), 0 )

        # Assign the correspondig value of the causal contribution
        cau = torch.zeros( (self.model.Nb, uv_.numel()) ).to(self.device) \
                + torch.nan
        for i in range( self.model.Nb ): cau[i][ inds == i ] = y[i][inds==i]

        return dev2np(cau).ravel(), dev2np(inds).ravel()


    def get_cau_val_nond( self ):
        '''get the causal contribution non-dim with the standard deviation of
        the source variable. 

            cau, sou, tar = get_cau_val_nond()

        Returns
            cau:    np.ndarray
                (N,) causal contribution from validation data set

            sou:    np.ndarray
                (N,) source variable, non-dim by std and mean

            tar:    np.ndarray
                (N,) target variable, non-dim by std and mean

            inds:   list[ Nb ]
                Indices that apply to each Nb for the causal contribution
        '''

        # Get the validation source and target
        qv_, uv_ = next(iter(self._va_l))

        # Compute the predicted value for each NN after sort them
        xd_s, _ = self.model.xd.sort()
        y = [ self.model.NNs[i]( qv_ ) for i in range( self.model.Nb ) ]
        
        # For the 1st NN, values larger than first extremum are NaN
        y[0][ y[0] > xd_s[0] ] = torch.nan
        
        # For the 2st-2nd to last NNs, values outside the extremum are Nan
        for i in range(1, self.model.Nb-1):
            y[i][ (y[i] < xd_s[i-1]) | (y[i] > xd_s[i])] = torch.nan

        # For the last NN, values smaller than first extremum are NaN
        y[-1][ y[-1] < xd_s[-1] ] = torch.nan
        
        y = torch.vstack( y ) # make 2d matrix from list

        # get the indices where the difference between u and cau are minimum.
        # For that, replace nan (non valid solutions, with crazy big values)
        inds = torch.argmin( torch.nan_to_num( (y - uv_)**2, nan=1e10 ), 0 )

        # Assign the correspondig value of the causal contribution
        cau = torch.zeros( uv_.size() ).to(self.device)
        for i in range( self.model.Nb ): cau[ inds == i ] = y[i][inds==i]

        return dev2np(cau).ravel(), dev2np(uv_).ravel(), \
                dev2np(qv_).ravel(), dev2np(inds).ravel()


    def get_cau_res_val( self ):
        '''get the causal contribution from validation dataset

            cau, res, sou, tar = get_cau_res_val()

        Returns
            cau:    np.ndarray
                (N,) causal contribution from validation data set
                
            res:    np.ndarray
                (N,) causal contribution from validation data set

            sou:    np.ndarray
                (N,) source variable

            tar:    np.ndarray
                (N,) target variable

            inds:   list[ Nb ]
                Indices that apply to each Nb for the causal contribution

        NOTE: variables have dimensions
        '''
        cau, sou, tar, inds = self.get_cau_val_nond()
        
        cau *= self.sS
        sou *= self.sS
        tar *= self.sT

        sou += self.mS
        tar += self.mT

        return cau, sou - cau, sou, tar, inds


    def get_cau_res_user( self, myq, myu ):
        '''get the causal/residual contribution from user data

            cau, res, inds = get_cau_res_val( myq, myu)

        Parameters
            myq:    np.ndarray
                Target variable with dimensions (any shape)

            myu:    np.ndarray
                Source variable with dimensions (myq.shape)

            norm:    bool
                if True, the input data is normalized, if False, it is assumed
                that the input is already normalized

        IMPORTANT: variables have to be shifted to account for time shift!

        Returns
            cau:    np.ndarray
                (myq.shape) causal contribution 
                
            res:    np.ndarray
                (myq.shape) causal contribution
        '''
        dset = self.getDataset( myq.ravel(), myu.ravel(), True, self.device )

        q_, u_ = dset.target, dset.source   # these are automatically ravel

        # Compute the value of phi for all q and for all functions
        xd_s, _ = self.model.xd.sort()
        y = [ self.model.NNs[i]( q_ ) for i in range( self.model.Nb ) ]
        
        # For the first NN, set to nans values when phi_c > phi
        y[0][ y[0] > xd_s[0] ] = torch.nan
        
        # Do the same for the 2 to Nb-2 NN
        for i in range(1, self.model.Nb-1):
            y[i][ (y[i] < xd_s[i-1]) | (y[i] > xd_s[i])] = torch.nan

        # For the last NN, set to nans phi_c < phi
        y[-1][ y[-1] < xd_s[-1] ] = torch.nan

        y = torch.vstack( y )
        # get the indices where the difference between u and cau are minimum.
        # For that, replace nan (non valid solutions, with crazy big values)
        inds = torch.argmin( torch.nan_to_num( (y - u_ )**2, nan=1e10 ), 0 )

        # Assign the correspondig value of the causal contribution
        cau = torch.zeros( myu.shape ).to(self.device)
        for i in range( self.model.Nb ): cau.ravel()[inds==i] = y[i][inds==i]
        
        cau = dev2np( cau )
        cau *= self.sS
        return cau, myu - cau, inds


    def get_funcs_user( self, myq  ):
        '''get the causal/residual contribution from user data

            funcs = get_funcs_user( myq )

        Parameters
            myq:    np.ndarray
                Target variable with dimensions (any shape)

        IMPORTANT: variables have to be shifted to account for time shift!

        Returns
            cau:    np.ndarray
                (myq.shape) causal contribution 
                
            res:    np.ndarray
                (myq.shape) causal contribution
        '''
        # Note that the source variable is dummy
        dset = self.getDataset( myq.ravel(), myq.ravel(), True, self.device )

        q_ = dset.target # these are automatically ravel

        # Compute the value of phi for all q and for all functions
        xd_s, _ = self.model.xd.sort()
        y = [ self.model.NNs[i]( q_ ) for i in range( self.model.Nb ) ]
        
        # For the first NN, set to nans values when phi_c > phi
        y[0][ y[0] > xd_s[0] ] = torch.nan
        
        # Do the same for the 2 to Nb-2 NN
        for i in range(1, self.model.Nb-1):
            y[i][ (y[i] < xd_s[i-1]) | (y[i] > xd_s[i])] = torch.nan

        # For the last NN, set to nans phi_c < phi
        y[-1][ y[-1] < xd_s[-1] ] = torch.nan

        y = torch.vstack( y )
        y *= self.sS

        return dev2np( y )

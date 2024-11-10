import numpy as np

import torch
from torch import nn
from torch.optim import lr_scheduler

from . import dsf, mikde
from .abstractNN import generalNN, dev2np

################################################################################
# Custom loss functions
################################################################################
class MI_Loss(nn.Module):
    '''Compute the loss function
            f1 * | var - cau |**2 + f2 * MI( cau, res )
    '''
    criterion = torch.nn.MSELoss(reduction='mean')

    def __init__(self, miobj: mikde.MutualInformation, f_mse: float | torch.Tensor,
                 f_mi: float | torch.Tensor ):
        super(MI_Loss, self).__init__()
        '''Compute the loss function
            f_mse * | var - cau |**2 + f_mi * MI( cau, res )

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


    def forward(self, y_pred, y_ ):
        '''Compute loss function

            res = obj( y_pred, y_, x_ )

        Parameters
            y_pred:  torch.Tensor
                prediction of NN (causal contribution)

            y_:  torch.Tensor
                target of NN ( variable to reconstruct )
        '''
        x_ = y_ - y_pred
        loss1 = self.criterion( y_pred, y_ )
        loss2 = self.mi( (y_pred-y_pred.mean()) / y_pred.std(), 
                         (x_ - x_.mean()) / x_.std() )

        return loss1*self.loss_fac + loss2.abs()*self.MI_fac, loss1, loss2


################################################################################
# bijective functions
################################################################################
class bijectiveNN( generalNN ):

    #criterion = torch.nn.MSELoss(reduction='mean')

    def __init__(self, model: str, num_layers: int, num_neurons: int | tuple,
                 run_gpu: bool = False, gpu_id: int= 0,
                 loadfrommodel: None | str= None):
        '''Initiliaze the bijective NN object

        Parameters:
            model:  str
                type of NN: DSF or DDSF

            num_layers: int
                number of layers for the NN

            num_neurons: int | tuple
                number of neurons per layer. If model = DDSF, it can be a tuple
                with the same number of elements as number of layers.

            run_gpu:    bool
                if True, model is run in gpu. Note that, if no gpus are available,
                the model will still run in cpu

            gpu_id:  int
                id of the gpu to run if cuda.is_available == True

            loadfrommodel:       None or str
                If a string, overwrite the model dimension and init a model with
                sizes of 'loadfrommodel' (only for DSF)

        '''
        generalNN.__init__( self, run_gpu, gpu_id )

        if model == "DSF":
            if loadfrommodel is not None:
                num_layers, num_neurons = self._get_DSF_modelsize( loadfrommodel )
                print( '   Ignoring inputs and initializing a DSF ->  '\
                      f'{num_layers} layers x {num_neurons} neurons/layer' )
            self.model = dsf.DSF( num_layers, num_neurons )

        elif model == "DDSF":
            if type(num_neurons) is int:
                self.model = dsf.DDSF( num_layers, num_neurons )
            else:
                raise ValueError( 'num_neurons must integer if model=DDSF' +
                                 f'(current value: {num_neurons})' )
        else:
            raise ValueError( f'model must be DSF or DDSF (current value: {model})' )


    def initTrainParams(self, **optparams ):
        '''initialize the variables and the optimizer

        Parameters
            **optparams: dic
                Parameters for Adam optimizer
        '''
        self.model.to( self.device )
        self.optimizer = torch.optim.Adam(self.model.parameters(), **optparams)


    def initMIParams( self, num_bins: int= 50, f_mi = 1., f_mse = .1 ):
        miobj, f_mi_n = generalNN._initMIParams( self, num_bins, f_mi = f_mi )

        # Overwrite citerion
        self.criterion = MI_Loss( miobj, f_mse, f_mi_n )


    def optimize( self, epochs: int = 10000, patience: int = 10,
                 lr_scheduler_dic={} ):
        '''train the NN

        Parameters
            epochs: int
                Number of training steps

            tol: int
                Early stop if no improvment is done in the validation set

        '''
        # The size of the batches is equal to the total length. We load the
        # variables here, because it is waaaay cheaper.
        q_ , u_  = next(iter(self._tr_l))
        qv_, uv_ = next(iter(self._va_l))

        #                   123456789ab_123456789ab_123456789ab_
        print( '\n      step    val loss    mse loss     mi loss' )
        #it, loss0, tol_min = 0, 1e15, False # init step and loss
        it, min_valid_loss, Counter_val = 0, 1000, 0

        scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                   verbose=True,
                                                   **lr_scheduler_dic)

        while ( it < epochs ):
            # Forward pass: Compute predicted y by passing x to the model
            u_pred = self.model( q_.view(1,-1) )

            # Zero gradients
            self.optimizer.zero_grad()

            # Compute and print loss
            loss,_,_ = self.criterion(u_pred, u_.view(1,-1))

            # perform a backward pass, and update the weights.
            loss.backward()
            self.optimizer.step()


            # Check if NN has improved
            if it % 100 == 0:
                self.model.eval()     # Optional when not using Model Specific layer

                u_pred = self.model( qv_.view(1,-1) )
                val_loss, e_loss, mi_loss  = self.criterion( u_pred,
                                                            uv_.view(1,-1) )

                # Update the learning rate based on validation loss
                scheduler.step(val_loss)

                if min_valid_loss > val_loss.item():
                    print( f'{it:10g}{val_loss.item():12.8f}',
                           f'{e_loss.item():12.8f}{mi_loss.item():12.8f}' )
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
        u_pred = self.model( qv_ )
        loss, _, _ = self.criterion(u_pred, uv_.view(1,-1))
        if Counter_val > patience:
            print(f'Optimization finished: Paticence ({patience}) achieved')
        else:
            print(f'Optimization finished: Maximum # steps ({epochs}) reached')

        print('Final error: ', loss.item())


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
        '''
        qv_, uv_ = next(iter(self._va_l))

        cau = self.model(qv_.view(1, -1) )

        return dev2np(cau).ravel(), dev2np(uv_).ravel(), dev2np(qv_).ravel()


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

        NOTE: variables have dimensions
        '''
        cau, sou, tar = self.get_cau_val_nond()

        cau *= self.sS
        sou *= self.sS
        tar *= self.sT

        sou += self.mS
        tar += self.mT

        return cau, sou - cau, sou, tar


    def get_cau_res_user( self, myq, myu )->tuple[np.ndarray,np.ndarray]:
        '''get the causal/residual contribution from user data

            cau, res = get_cau_res_val( myq, myu)

        Parameters
            myq:    np.ndarray
                Target variable with dimensions (any shape)

            myu:    np.ndarray
                Source variable with dimensions (myq.shape)

        IMPORTANT: variables have to be shifted to account for time shift!

        Returns
            cau:    np.ndarray
                (myq.shape) causal contribution

            res:    np.ndarray
                (myq.shape) causal contribution
        '''
        dset = self.getDataset( myq.ravel(), myu.ravel(), True, self.device )

        cau = dev2np( self.model( dset.target.view( 1, -1 ) ) ).reshape(
                myu.shape )

        cau *= self.sS

        return cau, myu - cau


    def get_cau_from_target( self, q ):
        return self.model( q.view(1,-1) ).detach().cpu().ravel()



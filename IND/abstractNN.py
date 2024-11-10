import re
from abc import ABC, abstractmethod
import numpy as np

from . import mikde

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from collections import OrderedDict
from pathlib import Path

def norm1dv(x):
    return (x - x.mean()) / x.std()

def norm_m_s(x,m_,s_):
    return (x - m_) / s_

def dev2np(x):
    return x.detach().cpu().numpy()

# Create a dataset class
class MyDataset(Dataset):
    '''Initialize a dataset for numpy input variables
    '''  
    def __init__(self, t_: np.ndarray, s_: np.ndarray,
                 device= torch.device("cpu") ):
        '''Initialize a dataset for numpy input variables
            dset = MyDataset( target, source, device )

        Parameters
            t_:     np.ndarray 
                array of the target variable -> size (N,) 

            s_:     np.ndarray 
                array of the source variable -> size (N,) 

            device:
                send the variables to device 
        '''
        self.source = torch.tensor( s_, dtype=torch.float32 ).to( device )
        self.target = torch.tensor( t_, dtype=torch.float32 ).to( device )
    
    def __getitem__(self, index):
        return self.target[index], self.source[index]
    
    def __len__(self):
        return len(self.source)


# Create abstract nn.module to define the model
class _NN(nn.Module):
    pass


class generalNN(ABC):

    model = _NN() # init abstract NN module

    def __init__(self, run_gpu: bool = False, gpu_id: int= 0):
        '''Initiliaze the general NN object

        Parameters:
            run_gpu:    bool
                if True, model is run in gpu. Note that, if no gpus are available,
                the model will still run in cpu

            gpu_id:  int
                id of the gpu to run if cuda.is_available == True

        '''
        if run_gpu:
            if torch.backends.mps.is_available():               # pyright: ignore (mac)
                self.device = torch.device("mps")
            elif torch.cuda.is_available():                     # linux
                self.device = torch.device(f"cuda:{gpu_id}")    
            else:
                self.device = torch.device("cpu")
        else: 
            self.device = torch.device("cpu")
        
        print( f"Using device: {self.device}" )


    def initTrainDset(self, T: np.ndarray, S: np.ndarray, dtlag: int, 
                        Nsamples: int= -1, Ntr: float= .7 ):
        '''initialize the variables and the optimizer

        Parameters
            T:  nd.array
                ( Nt, ... ) target variable 

            S:  nd.array
                ( Nt, ... ) source variable
            
            dtlag: int
                Time lag in steps
            
            Nsamples:   int
                Number of random samples that are taken for traning/validation
                from the original source variables.

            Ntr:        float
                (< 1.0) fraction of Nsamples for traning. Remaining are used for
                validation

        NOTE: q and u can have +1 dim, but the first dimension must
        always be time. Extra dimensions can correspond to different
        trajectories
        '''
        
        assert S.shape == T.shape, f'shape of target {T.shape} and source \
                {S.shape} does not match'

        # Roll the tar variable, that way, the target can be in the past
        # or in the future
        T = np.roll( T, -dtlag, axis=0 )

        # We remove the last/start steps, since they correspond to the 
        # rolled variable 
        if dtlag > 0:
            Snew, Tnew = S[:-dtlag], T[:-dtlag]
        elif dtlag < 0:
            Snew, Tnew = S[-dtlag:], T[-dtlag:]
        else:
            Snew, Tnew = S, T

        # Compute the mean and and standard deviation of the show data-set
        self.mS = Snew.mean()
        self.mT = Tnew.mean()

        self.sS = Snew.std()
        self.sT = Tnew.std()

        S_rav = Snew.flatten()
        T_rav = Tnew.flatten()

        del Snew, Tnew

        # Don't use all data for tranining / testing
        if Nsamples > 0: 
            inds = np.random.choice( T_rav.size, Nsamples )
            S_rav = S_rav[inds]
            T_rav = T_rav[inds]

        # Create data set. Move arrays to tensors
        dset = self.getDataset( T_rav, S_rav, norm=True, device= self.device )

        train_set, val_set = random_split( dset, [Ntr,1.-Ntr])

        self._tr_l = DataLoader(train_set, batch_size = len(train_set), 
                                shuffle=True  )
        self._va_l = DataLoader(val_set  , batch_size = len(val_set)  , 
                                shuffle=False )


    def _initMIParams( self, num_bins: int= 50, f_mi = .5 ):
        miobj =mikde.MutualInformation( num_bins = num_bins, sigma = 'scott', 
                            device = self.device)

        # Get the source variable to compute the entropy to normalize mutual
        # info
        _ , u_  = next(iter(self._tr_l))
        u_ = (u_ - u_.mean()) / u_.std()

        if miobj.sigma_method is not None:
            miobj._update_sigma( u_.shape[0], miobj.sigma_method )

        pdf_u, _ = miobj.marginalPdf( u_.view(-1,1))
        H_u = -torch.sum(pdf_u*torch.log2(pdf_u + miobj.epsilon) )

        return miobj, f_mi / H_u


    def getDataset( self, T_, S_, norm=True, device= torch.device("cpu") ):
        '''Get the target and source variables in numpy format and non-dim using
        the global mean and std.
        
        This function is called by other methods to compute the causal/residual
        contributions from user samples

        Parameters
            T_:     [np.ndarray]
                Target variable, can have any shape

            S_:     [np.ndarray]
                Target variable, can have any shape
        
        Returns
            dset:   [Dataset]
                pytorch Dataset to be called by other methods

        IMPORTANT: Time lag is not considered!! The source and target variables
        must be lagged appropiately
        '''
        nS = S_.ravel()
        nT = T_.ravel()

        # Normalize the whole data set
        if norm:
            nS = (nS - self.mS) / self.sS
            nT = (nT - self.mT) / self.sT

        # Create data set. Move arrays to tensors
        return MyDataset( nT, nS, device )


    def add_mean_std_to_model_dict(self ):
        '''Store the mean and standard deviation of the signals used for training.
        The variables are not accessed at any time during training, but they
        are useful to normalize the input for model testing. Specially if the
        model is saved with torch.save( model.state_dict() )

        '''
        self.model.mS = nn.Parameter( torch.Tensor( [self.mS,] ),   
                                     requires_grad=False)
        self.model.sS = nn.Parameter( torch.Tensor( [self.sS,] ), 
                                     requires_grad=False)
        self.model.mT = nn.Parameter( torch.Tensor( [self.mT,] ), 
                                     requires_grad=False)
        self.model.sT = nn.Parameter( torch.Tensor( [self.sT,] ), 
                                     requires_grad=False)


    def load_model( self, name_dict: str | Path | OrderedDict ):
        '''Load the model weights from dict'''

        # Add the mean and std as parameters in model (init with nan)
        self.mS, self.sS = np.nan, np.nan
        self.mT, self.sT = np.nan, np.nan
        self.add_mean_std_to_model_dict()

        # Load the weights and mean/std store in the state dict
        if isinstance( name_dict, str ) or isinstance( name_dict, Path ):
            self.model.load_state_dict( torch.load( name_dict , 
                                       map_location = self.device ) )
        elif isinstance( name_dict, OrderedDict ):
            self.model.load_state_dict( name_dict )
        else:
            raise ValueError( f'type(name_dict) = {type(name_dict)} \
            must be str, Path or OrderedDict' )

        self.model.to( self.device )

        # Assign mean and std to the object
        self.mS, self.sS = dev2np(self.model.mS)[0], dev2np(self.model.sS)[0]
        self.mT, self.sT = dev2np(self.model.mT)[0], dev2np(self.model.sT)[0]


    def _get_DSF_modelsize( self, name_dict ):

        ddd = torch.load( name_dict , map_location = self.device )
        layers = []
        for k in ddd.keys():
            dmy = re.search(r'layers.(\d+)\.', k)
            if dmy is not None: layers.append(dmy[0][7:-1])

        layers = list(dict.fromkeys(layers))

        num_neurons = ddd['layers.0.a0'].size(0)
        num_layers  = len(layers)

        return num_layers, num_neurons


    @abstractmethod
    def optimize( self, epochs: int, patience: int ):
        pass



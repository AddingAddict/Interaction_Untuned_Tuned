import numpy as np
import torch

import base_network as network

class RingNetwork(network.BaseNetwork):

    """
    Creates a network with n cell types on a ring representing an orientation dimension.
    The ring covers Nori=180 degrees.
    """

    def __init__(self, seed=0, Nori=8, n=None, NC=None):
        '''
        Construct a RingNetwork object.

        Parameters
        ----------
        seed : int
            Seed value for random number generator.
        Nori : int
            Number of sites representing preferred orientations in the network. Defaults to 8.
        n : int, optional
            Number of cell types. If not provided, defaults to the number of cell types in NC. If neither are provided, defaults to 1.
        NC : int or array-like, optional
            Number of cells of each cell type per site. If a single int is provided, it is used for all cell types. If not provided, the default is one cell per cell type per site.

        Attributes
        ----------
        n : int
            Number of cell types.
        NC : array-like
            Number of cells of each cell type per site.
        Nloc or Nori : int
            Number of sites in the network.
        Lori : int
            Length of the ring (180 degrees).
        NT : int
            Total number of cells per site.
        N : int
            Total number of neurons in the network (NT * Nloc).
        C_idxs : list of list
            List of indices for each cell type per site.
        C_all : list of array-like
            List of all indices for each cell type across all sites.
        Z : array-like
            Array of preferred orientations for each neuron in the network.
        '''
        self.Lori = 180

        self.Nori = int(Nori)
        self.Nloc = self.Nori

        super().__init__(seed=seed, n=n, NC=NC, Nloc=self.Nloc, profile="wrapgauss")

        self.set_Z()

    def set_Z(self):
        '''
        Set the preferred orientations for each neuron in the network.
        '''
        Z = np.array(np.unravel_index(np.arange(self.Nloc),(self.Nori,))).astype(float)
        Z *= self.Lori/self.Nori
        Z = np.repeat(Z,self.NT,axis=1)

        self.Z = Z[0]

    def get_ori_dist(self,vis_ori=None,byloc=False):
        '''
        Get the distance between cells' preferred orientations and the visual stimulus orientation.
        
        Parameters
        ----------
        vis_ori : float, optional
            Visual stimulus orientation.
        byloc : bool
            If True, calculate the distance once per site. If False, calculate the distance for all cells.
        
        Returns
        -------
        array-like
            1D Array of distances between the preferred orientations of cells and the visual stimulus orientation.
        '''
        if vis_ori is None:
            vis_ori = 0.0
        if byloc:
            idxs = slice(0,self.N,self.NT)
        else:
            idxs = slice(0,self.N)
        ori_dist = network.make_periodic(np.abs(self.Z[idxs] - vis_ori),self.Lori/2)
        return ori_dist

    def get_ori_diff(self,byloc=False):
        '''
        Get the difference between the preferred orientations of cells.
        
        Returns
        -------
        array-like
            2D Array of differences between the preferred orientations of cell pairs.
        '''
        if byloc:
            idxs = slice(0,self.N,self.NT)
        else:
            idxs = slice(0,self.N)
        ori_diff = network.make_periodic(np.abs(self.Z[idxs] - self.Z[idxs][:,None]),self.Lori/2)
        return ori_diff

    def get_oriented_neurons(self,delta_ori=22.5,vis_ori=None):
        '''
        Get the indices of neurons that are within a certain distance from the visual stimulus orientation.
        
        Parameters
        ----------
        delta_ori : float
            Distance from the visual stimulus orientation to consider a neuron as "oriented".
        vis_ori : float, optional
            Visual stimulus orientation.
            
        Returns
        -------
        array-like
            1D Array of indices of neurons that are within delta_ori from the visual stimulus orientation.
        '''
        ori_dist = self.get_ori_dist(vis_ori)
        return np.where(ori_dist < delta_ori)

    def generate_full_vector(self,Sori,kernel="basesubwrapgauss",byloc=True,vis_ori=None):
        '''
        Generate vector of the kernel applied to the distance between cells' preferred orientations and the visual stimulus orientation.
        
        Parameters
        ----------
        Sori : float
            Kernel width.
        kernel : str
            Kernel type. Defaults to "basesubwrapgauss".
        byloc : bool
            If True, calculate the kernel once per site. If False, calculate the kernel for all cells.
        vis_ori : float, optional
            Visual stimulus orientation.
            
        Returns
        -------
        array-like
            1D Array of the kernel applied to the distance between cells' preferred orientations and the visual stimulus orientation.
        '''
        ori_dist = self.get_ori_dist(vis_ori=vis_ori,byloc=byloc)
        full_vector = network.apply_kernel(ori_dist,Sori,self.Lori,kernel=kernel)
        return full_vector

    def generate_full_kernel(self,Sori,kernel="wrapgauss",byloc=True):
        '''
        Generate matrix of the kernel applied to the difference between the preferred orientations of cells.
        
        Parameters
        ----------
        Sori : float
            Kernel width.
        kernel : str
            Kernel type. Defaults to "wrapgauss".
        byloc : bool
            If True, calculate the kernel once per site. If False, calculate the kernel for all cells.
            
        Returns
        -------
        array-like
            2D Array of the kernel applied to the difference between the preferred orientations of cell pairs.
        '''
        ori_diff = self.get_ori_diff(byloc=byloc)
        full_kernel = network.apply_kernel(ori_diff,Sori,self.Lori,self.Lori/self.Nori,kernel=kernel)
        full_kernel /= np.sum(full_kernel,1)[:,None]
        return full_kernel

    def generate_full_rec_conn(self,WMat,VarMat,SoriMat,K,baseprob=0,rho=0,return_mean_var=False):
        '''
        Generate the components needed to create the full recurrent connectivity matrix for the network.
        
        Parameters
        ----------
        WMat : array-like
            nxn Matrix of mean synaptic efficacies between cell types.
        VarMat : array-like
            nxn Matrix of the variances of the synaptic efficacies between cell types.
        SoriMat : array-like
            nxn Matrix of the kernel widths for the synaptic efficacies between cell types.
        K : float or array-like
            Mean of in-degree per connection type. If a single float is provided, K is set as the mean in-degree of the first cell type and the in-degree of cell type i>0 is scaled by NC[i]/NC[0]. If an nxn 2D array is provided, the elements are used as the in-degree for each connection type.
        baseprob : float
            Baseline probability of connection. Defaults to 0. A value of 1 indicates fully connected unstructured connectivity.
        rho : float or array-like
            Correlation of the binary connections. If a single float is provided, rho is set as the correlation for all connection types. If an nxn 2D array is provided, the elements are used as the correlation for each connection type.
        return_mean_var : bool
            If True, return the mean and variance of the synaptic efficacies for each connection type.
            
        Returns
        -------
        array-like
            2D Array of the binary connections between all cell pairs.
        array-like
            2D Array of the mean synaptic efficacies between all cell pairs.
        array-like
            2D Array of the variances of the synaptic efficacies between all cell pairs.
        '''
        WKern = [[None]*self.n]*self.n
        W_mean_full = np.zeros((self.N,self.N),np.float32)
        W_var_full = np.zeros((self.N,self.N),np.float32)

        for pstC in range(self.n):
            pstC_idxs = self.C_idxs[pstC]
            for preC in range(self.n):
                preC_all = self.C_all[preC]
                
                if np.isclose(WMat[pstC,preC],0.): continue    # Skip if no connection of this type

                if baseprob==1:
                    W = np.ones((self.Nloc,self.Nloc)) / self.Nloc
                else:
                    W = baseprob/self.Nloc + (1-baseprob)*self.generate_full_kernel(SoriMat[pstC,preC])

                WKern[pstC][preC] = W

                for pst_loc in range(self.Nloc):
                    pst_idxs = pstC_idxs[pst_loc]

                    W_mean_full[pst_idxs,preC_all] = WMat[pstC,preC]
                    W_var_full[pst_idxs,preC_all] = VarMat[pstC,preC]

        if (np.isscalar(rho) and np.isclose(rho,0)) or np.all(np.isclose(rho,0)):
            C_full = self.generate_sparse_rec_conn(WKern=WKern,K=K)
        else:
            C_full = self.generate_corr_sparse_rec_conn(WKern=WKern,K=K,rho=rho)

        if return_mean_var:
            return C_full, W_mean_full,W_var_full
        else:
            return C_full

    def generate_full_input(self,HVec,VarVec,SoriVec,baseinp=0,vis_ori=None):
        '''
        Generate the components needed to create the full input vector for the network.
        
        Parameters
        ----------
        HVec : array-like
            1D Array of the mean input for each cell type.
        VarVec : array-like
            1D Array of the variance of the input for each cell type.
        SoriVec : array-like
            1D Array of the kernel widths for the input for each cell type.
        baseinp : float
            Baseline input. Defaults to 0. A value of 1 indicates unstructured input.
        vis_ori : float, optional
            Visual stimulus orientation.
            
        Returns
        -------
        array-like
            1D Array of the mean input to each cell.
        array-like
            1D Array of the variance of the input to each cell.
        '''
        H_mean_full = np.zeros((self.N),np.float32)
        H_var_full = np.zeros((self.N),np.float32)

        for pstC in range(self.n):
            pstC_all = self.C_all[pstC]
            NpstC = self.NC[pstC]

            if np.isclose(HVec[pstC],0.): continue    # Skip if no connection of this type

            if baseinp==1:
                H = np.ones((self.Nloc))
            else:
                H = baseinp + (1-baseinp)*self.generate_full_vector(SoriVec[pstC],vis_ori=vis_ori)

            H_mean_full[pstC_all] = HVec[pstC]*np.repeat(H,NpstC)
            H_var_full[pstC_all] = VarVec[pstC]*np.repeat(H,NpstC)

        return H_mean_full,H_var_full

    def generate_M(self,W,SWori,K,baseprob=0,rho=0):
        '''
        Generate the full recurrent connectivity matrix for the network.
        
        Parameters
        ----------
        W : array-like
            nxn Matrix of mean synaptic efficacies between cell types.
        SWori : array-like
            nxn Matrix of the kernel widths for the synaptic efficacies between cell types.
        K : float or array-like
            Mean of in-degree per connection type. If a single float is provided, K is set as the mean in-degree of the first cell type and the in-degree of cell type i>0 is scaled by NC[i]/NC[0]. If an nxn 2D array is provided, the elements are used as the in-degree for each connection type.
        baseprob : float
            Baseline probability of connection. Defaults to 0. A value of 1 indicates fully connected unstructured connectivity.
        rho : float or array-like
            Correlation of the binary connections. If a single float is provided, rho is set as the correlation for all connection types. If an nxn 2D array is provided, the elements are used as the correlation for each connection type.
            
        Returns
        -------
        array-like
            2D Array of the recurrent weight matrix between all cell pairs.
        '''
        C_full, W_mean_full,W_var_full = self.generate_full_rec_conn(W,np.zeros((self.n,self.n)),
            SWori,K,baseprob,rho,True)
        return C_full*(W_mean_full+np.random.normal(size=(self.N,self.N))*np.sqrt(W_var_full))

    def generate_H(self,H,SHori,baseinp=0,vis_ori=None):
        '''
        Generate the full input vector for the network.
        
        Parameters
        ----------
        H : array-like
            1D Array of the mean input for each cell type.
        SHori : array-like
            1D Array of the kernel widths for the input for each cell type.
        baseinp : float
            Baseline input. Defaults to 0. A value of 1 indicates unstructured input.
        vis_ori : float, optional
            Visual stimulus orientation.
            
        Returns
        -------
        array-like
            1D Array of the input to each cell.
        '''
        H_mean_full,H_var_full = self.generate_full_input(H,np.zeros((self.n)),SHori,baseinp,vis_ori)
        return H_mean_full+np.random.normal(size=(self.N))*np.sqrt(H_var_full)

    def generate_disorder(self,W,SWori,H,SHori,K,baseinp=0,baseprob=0,rho=0,vis_ori=None):
        '''
        Generate the full recurrent connectivity matrix and input vector for the network.
        
        Parameters
        ----------
        W : array-like
            nxn Matrix of mean synaptic efficacies between cell types.
        SWori : array-like
            nxn Matrix of the kernel widths for the synaptic efficacies between cell types.
        H : array-like
            1D Array of the mean input for each cell type.
        SHori : array-like
            1D Array of the kernel widths for the input for each cell type.
        K : float or array-like
            Mean of in-degree per connection type. If a single float is provided, K is set as the mean in-degree of the first cell type and the in-degree of cell type i>0 is scaled by NC[i]/NC[0]. If an nxn 2D array is provided, the elements are used as the in-degree for each connection type.
        baseinp : float
            Baseline input. Defaults to 0. A value of 1 indicates unstructured input.
        vis_ori : float, optional
            Visual stimulus orientation.
        '''
        self.M = self.generate_M(W,SWori,K,baseprob,rho)
        self.H = self.generate_H(H,SHori,baseinp,vis_ori=vis_ori)

    def generate_tensors(self):
        '''
        Generate PyTorch tensors for the cell-type indices, recurrent connectivity matrix, and input vector.
        '''
        self.C_conds = []
        for cidx in range(self.n):
            this_C_cond = torch.zeros(self.N,dtype=torch.bool)
            this_C_cond = this_C_cond.scatter(0,torch.from_numpy(self.C_all[cidx]),
                torch.ones(self.C_all[cidx].size,dtype=torch.bool))
            self.C_conds.append(this_C_cond)

        self.M_torch = torch.from_numpy(self.M.astype(dtype=np.float32))
        self.H_torch = torch.from_numpy(self.H.astype(dtype=np.float32))
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print("Using",device)

        for cidx in range(self.n):
            self.C_conds[cidx] = self.C_conds[cidx].to(device)
        self.M_torch = self.M_torch.to(device)
        self.H_torch = self.H_torch.to(device)

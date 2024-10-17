"""Dedenser class for downsampling chemical point clouds.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import random
from sklearn.cluster import HDBSCAN
from scipy.spatial import ConvexHull
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
import point_cloud_utils as pcu
import alphashape
import matplotlib.pyplot as plt


class Dedenser( ):
    """Class handles chemical point cloud downsampling.

    `Dedenser` takes a user defined target percent to
    attempt downsampling of a given chemical point cloud.

    Parameters
    ----------
    data : array-like of shape (N, 3)
        Initial chemical point cloud to be downsampled.

    target : float, default=None
        Target percentage for the chemical point cloud to
        be downsampled to.
    
    random_seed : int, default=1
        Random seed to be used by point cloud utils.    

    alpha : bool, default=False
        Determine if alphashapes (concave hulls) are used
        to estimate cluster volumes rather than convex hulls.

    epsilon : float, default=0.0
        Distance threshold for cluster merging used by HDBSCAN.

    d_weight : float or None, default=None
        Allows users to weight cluster membership targets based
        on the relative density of clusters.  High weights will
        favor the retention of members in denser clusters.

    v_weight : float or None, default=None
        Allows users to weight cluster membership targets based
        on the relative volume of clusters.  High weights will
        favor the retention of members in spatially large clusters.

    min_size : int, default=15
        The min_size parameter for HDBSCAN.

    strict : bool, default=False
        Determine if clusters with target points to be kept
        of 0 are to be floored at 1 (`False`) or retained
        as 0 (`True`).

    show : bool, default=False
        When set to True, will display clusters (colored) and 
        noise (black) after HDBSCAN clustering.

    Attributes
    ----------
    n_target : int
        Target number of points based on the percentage.

    r_target : int
        Remaining number of points to select.

    keep_list : list
        List of indexes to keep after downsampling.

    state : str
        The current state of downsampling: 'thinner' or 'done'.

    clusters : numpy.ndarray
        Array of cluster labels after clustering.

    vol : float
        The volume of the convex hull of all clusters.

    Methods
    -------
    start()
        Performs clustering and calculates cluster volumes.

    check_state()
        Checks the state of the downsampling process.

    thinner()
        Performs downsampling when the target percentage is not achieved.

    choice(indexs, targ)
        Randomly selects indexes from a given list.

    downsample()
        The main method that initiates the downsampling process.
    """
    def __init__(self,data=None, target=.5, random_seed=1, alpha=False, 
                 min_size = 5, d_weight = None, v_weight = None, epsilon=0.0,
                 strict=False, show=False,):
        if target in (0, 1):
            raise ValueError(f'Target can not be {target}.')
        if target > 1 or target < 0:
            raise ValueError(f'Target must be between 0 & 1.\
                               User provided {target}')
        self.n_target = target * len(data)
        self.data = np.array(data)
        if data.shape[1] != 3:
            raise ValueError("Data must have shape (N, 3).")
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError("Data must contain numerics.")
        self.Strict = strict
        self.random_seed = random_seed
        self.alpha = alpha
        self.keep_list = []
        self.state = ''
        self.clusters = []
        self.vol = 0
        self.epsilon = epsilon
        self.vols = []
        self.r_target = None
        self.min_size = min_size
        self.d_weight = d_weight
        self.v_weight = v_weight
        self.Weighted = False
        if self.v_weight != None or self.d_weight != None:
            self.Weighted = True
        self.Show = show

    def start(self):
        """Performs clustering and calculates cluster volumes."""
        hdbscan = HDBSCAN(min_cluster_size=self.min_size,cluster_selection_epsilon=self.epsilon)
        self.keep_list = list(self.keep_list)
        self.clusters = hdbscan.fit(self.data).labels_
        c = 0
        for point in self.clusters:
            if point == -1:
                self.keep_list.append(c)
            c += 1
        self.keep_list = list(set(self.keep_list))
        for cluster in range(0, max(self.clusters)+1):
            c_points = [self.data[i] for i, x in enumerate(self.clusters) if x == cluster]
            if self.alpha:
                #alpha = alphashape.optimizealpha(c_points)
                hull = alphashape.alphashape(c_points, 0.15)
                self.vol += abs(hull.volume)
                self.vols.append(abs(hull.volume))
            else:
                hull = ConvexHull(c_points)
                self.vol += hull.volume
                self.vols.append(hull.volume)

        if self.Show:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('UMAP_1')
            ax.set_ylabel("UMAP_2")
            ax.set_zlabel("UMAP_3")
            c_points = np.array([self.data[i] for i, x in enumerate(self.clusters) if x == -1])
            if len(c_points) != 0:
                ax.scatter(c_points[:, 0], c_points[:, 1], c_points[:, 2],s=.3,c='k')
            for cluster in range(0, max(self.clusters)+1):
                c_points = np.array([self.data[i] for i, x in enumerate(self.clusters) if x == cluster])
                ax.scatter(c_points[:, 0], c_points[:, 1], c_points[:, 2],s=.3)
            plt.tight_layout()
            #plt.savefig('data/initial_cloud.svg',dpi=1000)
            plt.show()

    def check_state(self):
        """Cheks the state after using HDBSCAN.
        
        Downsamples the entire chemical cloud if HDBSCAN selected too much noise
        given the target or continues with the dedensing algorithm otherwise.
        """
        if self.n_target*.95 < len(self.keep_list):#can make a variable threshold

            self.state = 'noisy'
        else:
            self.state = 'thinner'
            self.r_target = self.n_target - len(self.keep_list)

    def thinner(self):
        """Performs downsampling when the target percentage is not achieved.

        Calculates the number of points to be removed per cluster based on their volumes.
        Performs downsampling using Poisson disk sampling for larger clusters.
        """
        c_lens, c_pointses, c_is, cds = [], [], [], []
        for cluster in range(0,max(self.clusters)+1):
            c_index = [i for i, x in enumerate(self.clusters) if x == cluster]
            c_is.append(c_index)
            c_lens.append(len(c_index))
            c_points = [self.data[i] for i, x in enumerate(self.clusters) if x == cluster]
            cpl = []
            for point in c_points:
                cpl.append(list(point))
            c_pointses.append(cpl)
            cds.append(len(c_index)/self.vols[cluster])
            #print(np.mean(np.array(c_points)[:,0]),np.mean(np.array(c_points)[:,1]),np.mean(np.array(c_points)[:,2]))
            #print(len(c_index)/self.vols[cluster], self.vols[cluster], len(c_index))
        #estimate how many points to be removed per cluster
        Check_Targs = True
        while Check_Targs:
            Check_Targs, c_is, c_pointses, cds = self.manage_targets(c_is,c_pointses,cds)
        o_targ = 0
        for cvol, cind, points, cd in zip(self.vols,c_is,c_pointses,cds):
            targ = round(self.r_target*cvol/sum(self.vols))
            if self.Weighted:
                if self.d_weight != None:
                    w = self.d_weight
                    d1 = np.exp(w * (cd/sum(cds) - 1))
                    dt = sum([np.exp(w * (cdn/sum(cds) - 1)) for cdn in cds])
                    targ = round(self.r_target*(d1/dt))
                else:
                    w = self.v_weight
                    v1 = np.exp(w * (cvol/sum(self.vols) - 1))
                    vt = sum([np.exp(w * (cvols/sum(self.vols) - 1)) for cvols in self.vols])
                    targ = round(self.r_target*(v1/vt))
            if targ == 0:
                pass
            if targ < 15:
                if targ <= 0:
                    if self.Strict:
                        targ = 0
                    else:
                        o_targ += 1
                        targ = 1
                gmm = GaussianMixture(n_components=1, random_state=0)
                gmm.fit(points)
                #get the centroid of the fitted GMM
                centroid = gmm.means_[0]
                center = np.argmin(cdist(points, [centroid]))
                if targ == 1:
                    inds = [center]
                elif targ != 0:
                    inds = self.choice(cind,targ)
                    if not (center in inds):
                        ri = random.randint(0, len(inds) - 1)
                        inds = list(inds)
                        del inds[ri]
                        inds.append(center)
                if targ != 0:
                    for ind in inds:
                        self.keep_list = list(self.keep_list)
                        self.keep_list.append(cind[ind])
            elif targ != 0:
                target_radius = -1
                idx = pcu.downsample_point_cloud_poisson_disk(np.array(points), target_radius,targ,
                                                              random_seed=self.random_seed,
                                                              sample_num_tolerance=0.01)
                self.keep_list = list(self.keep_list)
                for ind in idx:
                    self.keep_list.append(cind[ind])
            self.keep_list = np.array(self.keep_list)

    def manage_targets(self,c_is,c_pointses,cds):
        """Checks for clusters with fewer points than their target value.

        Updates the remaining targets and clusters by keeping all points from clusters
        with fewer points than their target, and removing them from the running list of clusters.
        """
        New_Targs = False
        n_vols, n_is, n_pointses, n_cds = [], [], [], []
        for cvol, cind, points, cd in zip(self.vols,c_is,c_pointses,cds):
            targ = round(self.r_target*cvol/sum(self.vols))
            if self.Weighted:
                if self.d_weight != None:
                    w = self.d_weight
                    d1 = np.exp(w * (cd/sum(cds) - 1))
                    dt = sum([np.exp(w * (cdn/sum(cds) - 1)) for cdn in cds])
                    targ = round(self.r_target*(d1/dt))
                else:
                    w = self.v_weight
                    v1 = np.exp(w * (cvol/sum(self.vols) - 1))
                    vt = sum([np.exp(w * (cvols/sum(self.vols) - 1)) for cvols in self.vols])
                    targ = round(self.r_target*(v1/vt))
            if targ > len(cind):
                New_Targs = True
                self.keep_list = list(self.keep_list)
                for point in cind:
                    self.keep_list.append(point)
                    self.keep_list = list(set(self.keep_list))
            else:
                n_vols.append(cvol)
                n_is.append(cind)
                n_pointses.append(points)
                n_cds.append(cd)
        if New_Targs:
            self.vols = n_vols
            c_is = n_is
            c_pointses = n_pointses
            cds = n_cds
        self.r_target = self.n_target - len(self.keep_list)
        return New_Targs, c_is, c_pointses, cds
        

    def choice(self,indexs,targ):
        """Helper method for numpy randomchoice for the sake of readability.
        """
        return np.random.choice(len(indexs),targ)

    def downsample(self):
        """The main method that initiates and runs the downsampling process.

        Calls the `start()` method to perform clustering and calculate cluster volumes
        and then checks the state of the downsampling process with `check_state()`.
        If the state is `'noisy'`, the noise from HDBSCAN is downsampled based on its volume
        prior to the downsampling of clusters using the `thinner()` method.
        If the state is `'thinner'`, performs downsampling using the `thinner()` method.
        After `thinner()` is run, it returns the list of point cloud indexes to be kept.

        Returns
        -------
        keep_list : numpy.ndarray
            List of indexes to be kept after downsampling.

        """
        self.start()
        print(f"Target of {int(round(self.n_target))} molecules")
        self.check_state()
        if self.state == 'noisy':
            noise = self.data[np.array(self.keep_list)]
            if self.alpha:
                noise_v = abs(alphashape.alphashape(noise, 0.15).volume)
            else:
                noise_v = ConvexHull(noise).volume
            no_targ = round(noise_v/(noise_v+sum(self.vols)) * self.n_target * .8)
            idx = pcu.downsample_point_cloud_poisson_disk(np.array(noise), -1,no_targ,
                                                              random_seed=self.random_seed,
                                                              sample_num_tolerance=0.01)
            self.keep_list = np.array(self.keep_list)
            self.keep_list = self.keep_list[np.array(idx)]
            self.keep_list = list(set(self.keep_list))
            self.r_target = self.n_target - len(self.keep_list)
        self.thinner()
        print("Downsampled to {0} molecules".format(len(self.keep_list)))
        if self.Show:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('UMAP_1')
            ax.set_ylabel("UMAP_2")
            ax.set_zlabel("UMAP_3")
            c_points = np.array([self.data[i] for i, x in enumerate(self.clusters) if x == -1 and i in self.keep_list])
            if len(c_points) != 0:
                ax.scatter(c_points[:, 0], c_points[:, 1], c_points[:, 2],s=.3,c='k')
            for cluster in range(0, max(self.clusters)+1):
                c_points = np.array([self.data[i] for i, x in enumerate(self.clusters) if x == cluster and i in self.keep_list])
                if len(c_points) != 0:
                    ax.scatter(c_points[:, 0], c_points[:, 1], c_points[:, 2],s=.3)
                else:
                    ax.scatter([], [], [],s=.3)
            plt.tight_layout()
            #plt.savefig('data/downsampled_cloud.svg',dpi=1000)
            plt.show()
        return list(set(self.keep_list))

import numpy as np
from scipy import stats
import os
import pandas as pd
import itertools
import copy
import paddy
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import random
import subprocess
import re


from argparse import ArgumentParser
import sys
import os
import logging
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem.Descriptors import qed
from pathlib import Path

sys.path.append( 'DeepDL-master/DeepDL-master/src')
from models import RNNLM, GCNModel
import utils as UTILS

import dedenser

np.random.seed(100)

print('imports done')
class Numeric_Assistant(object):
    def __init__(self,observations,repeats=True,raw_responce=False):
        self.observations = observations
        self.repeats = repeats
        self.obs_clone = np.array(copy.deepcopy(observations))
        self.black_list = []
        self.norm_cloner()
        self.raw_responce = raw_responce
        print(observations)
    def sample(self,inputs):
        temp = []
        for i in inputs:
            temp.append(i[0])
        inputs = temp
        distance_list = []
        distance_vectors = []
        c = 0
        for i in range(len(self.obs_clone[0])):#get distances for every param combo
            distance_vectors.append(np.linalg.norm(self.norm_clone[:-1,c] - inputs))
            c += 1
        distance_list.append(distance_vectors)#append distances for each domain/param
        score_list = []
        if not self.repeats:#doesnt have error handling for equal distances
            self.black_list = []
            #print("distance list:",distance_list)
            #print("norm clone", self.norm_clone)
            #print("distance list len:",len(distance_list[0]))
            #print("dist len",len(distance_list[0]))
            if len(distance_list[0]) > 1:
                for i in distance_list:
                    repeats = i.count(min(i))
                    if repeats == 1:
                        if self.raw_responce:
                            score_list.append(self.obs_clone[-1][np.argmin(i)])
                        else:
                            score_list.append(self.norm_clone[-1][np.argmin(i)])
                        self.black_list.append(np.argmin(i))
                        self.out_params = self.obs_clone[:-1,np.argmin(i)]#descrete conditions sampled
                    if repeats > 1:
                        d = 0
                        r_list = []
                        for j in i:
                            if j == min(i):
                                r_list.append(d)
                            d +=1
                        rc = random.choice(r_list)
                        if self.raw_responce:
                            score_list.append(self.obs_clone[-1][rc])
                        else:
                            score_list.append(self.norm_clone[-1][rc])
                        self.black_list.append(rc)
                        self.out_params = self.obs_clone[:-1,rc]#descrete conditions sampled
            if len(distance_list[0]) == 1:
                if self.raw_responce:
                    score_list.append(self.obs_clone[-1][0])
                else:
                    score_list.append(self.norm_clone[-1][0])
                self.black_list.append(0)
            self.norm_cloner()#renormalizes after sampling and removes observations
            #distance_list is values of selection out of vector of posibilites
            #print("Black list len:", len(self.black_list))
            #print("# Observations:", len(self.obs_clone[0]))
        else:
            for i in distance_list:
                if self.raw_responce:
                    score_list.append(self.obs_clone[-1][np.argmin(i)])
                else:
                    score_list.append(self.norm_clone[-1][np.argmin(i)])
            self.out_params = self.obs_clone[:-1,np.argmin(i)]
        return(score_list[0])
    def min_max(self, x, minV, maxV):
        if minV != maxV:
            x = x
            minV = minV
            maxV = maxV
            return((x-minV)/(maxV-minV))
        else:
            return(1)
    def norm_cloner(self):
        #clones observations with normalized values between 0 and 1
        self.norm_clone =[]
        if self.repeats:
            for row in self.observations:
                print(row)
                minV, maxV = min(row), max(row)
                temp =[]
                for value in row:
                    normed = self.min_max(value, minV, maxV)
                    temp.append(normed)
                self.norm_clone.append(temp)
        else: 
            self.obs_cloner()
            if len(self.obs_clone[0]) > 0: 
                for row in self.obs_clone:
                    minV, maxV = min(row), max(row)
                    temp =[]
                    for value in row:
                        normed = self.min_max(value, minV, maxV)
                        temp.append(normed)
                    self.norm_clone.append(temp)
        self.norm_clone = np.array(self.norm_clone)    
    def obs_cloner(self):
        #uses black list to ommit prior observations
        obs_clone2 = []
        c2 = 0
        for i in self.obs_clone:
            obs_clone2.append([])
            c = 0
            for j in i:
                if c in self.black_list:
                    pass
                else:
                    obs_clone2[c2].append(j)
                c += 1
            c2 += 1
        self.obs_clone = np.array(obs_clone2)


data = pd.read_csv('ZINC_bi.csv')
smiles = data.values[:,0]
logp, smr, mw, topopsa, bertzct = data['SLogP'].values, data['SMR'].values, data['MW'].values, data['TopoPSA'].values, data['BertzCT'].values

sl = np.arange(len(smiles))

observations = []

for i,j,k,l,m,n in zip(logp,smr,mw,topopsa,bertzct,sl):
	observations.append([float(i),float(j),float(k),float(l),float(m),n])


observations = np.array(observations).T

paddy_sampler = Numeric_Assistant(observations=observations,repeats=True,raw_responce=True)


mins = []
maxs = []

for param in observations[:-1]:
    mins.append(min(param))
    maxs.append(max(param))

def min_max(x, minV, maxV):
    if minV != maxV:
        x = x
        minV = minV
        maxV = maxV
        return((x-minV)/(maxV-minV))
    else:
        return(1)

model_path = 'DeepDL-master/DeepDL-master/test/result/rnn_pubchem_worlddrug'
device = 'cpu'
model_config_file = os.path.join(model_path, 'config.yaml')
config = OmegaConf.load(model_config_file)
model_architecture = config.model.model # RNNLM or GCNModel

model = RNNLM.load_model(model_path, device)

def sample(input_v):
    global bv
    global bsmile
    smile = paddy_sampler.sample(input_v)
    print(smiles[int(smile)])
    score = model.test(smiles[int(smile)])
    if score > bv:
        bv = score
        bsmile = smiles[int(smile)]
    print(score)
    return(score)

def dummy_sample(input_v):
    return('throws error')

norm_param = paddy.PaddyParameter(param_range=[0,1,.01],
                                    param_type='continuous',
                                    limits=[0,1], gaussian='default',
                                    normalization = False)

class chemical_space(object):
    def __init__(self):
        self.a = norm_param
        self.b = norm_param
        self.c = norm_param
        self.d = norm_param
        self.e = norm_param

rs = chemical_space()


dd = np.load('ZINC_bi_d10.npy')

sl = []
bl = []
ml = []
for repeat in range(25):
    bv = 0
    bsmile = ''
    bind = 0
    paddy_sampler = Numeric_Assistant(observations=observations,repeats=False,raw_responce=True)
    try:    
        runner = paddy.PFARunner(space=rs,eval_func=dummy_sample,paddy_type='population',rand_seed_number=99,yt=5,Qmax=10,r=.1,iterations=1)
        runner.run_paddy(file_name='temp')
    except:
        print()
    runner = paddy.utils.paddy_recover('temp')
    print('recover')
    paddy_sampler = Numeric_Assistant(observations=observations,repeats=False,raw_responce=True)
    for seed in range(99):
        runner.seed_fitness[seed] = model.test(smiles[dd[seed]])
        for param in range(5):
             runner.seed_params[seed][param][0] = min_max(observations[param][dd[seed]],mins[param],maxs[param])
        paddy_sampler.sample(runner.seed_params[seed])
    runner.eval_func = sample
    runner.iterations=3
    runner.recover_run()
    ml.append(max(runner.seed_fitness))
    sl.append(bsmile)
    bl.append(np.argmax(runner.seed_fitness))

np.save('dedenser_ml',ml)
np.save('dedenser_sm',sl)
np.save('dedenser_bl',bl)


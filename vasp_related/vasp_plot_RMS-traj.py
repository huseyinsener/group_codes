#!/usr/bin/env python3
#,print_function

import os,time,re,math,argparse #modify this to specify specific parts that are used.
from sys import exit,stdout,argv,version_info
from copy import deepcopy as dc
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
#from matplotlib import interactive
from numpy import linalg as la
from numpy import array ,zeros,matrix
from numpy import cos, sin, sqrt
import numpy as np
from ase import Atoms
import ase.io
from ase.visualize import view
from ase.geometry import find_mic  #min dist through PBC using the minimum-image covnention
#from joblib import Parallel, delayed
#import multiprocessing
from multiprocessing import Pool,Process,Value,Array
#from multiprocessing.sharedctypes import Value, Array
#import threading
#import numba
#from numba import jit,autojit, prange,njit #This requires module load gcc/4.8.3
import concurrent  #Can be installed using pip install futures.
from concurrent.futures import Executor,ThreadPoolExecutor
from os import popen#,popen4 #popen4 is not compatible with python3
from subprocess import Popen,PIPE # as popen # check_output
#from scipy.spatial.distance import cdist
#from scipy import signal

import matplotlib.ticker as ticker
from matplotlib.widgets import TextBox
import os.path

from scipy.stats import linregress
#import statsmodels.api as sm
#from sklearn.linear_model import LinearRegression

from ase.geometry.analysis import Analysis

#global minDist,deepcopy,dist
global Z, e, kb,angtocm
# ------------------- sener modified this starting from here --------------------------

#Z={'Na':1,'Li':1,'Mg':2,'Ca':2,'K':1,'H':1,'Sr':2} #ionic charge dict.
Z={'Na':1,'Li':1,'Mg':2,'Ca':2,'K':1,'H':1,'Sr':2,'S':1,'P':1,'Nb':1,'O':1} #ionic charge dict.

# to here <----------------------------------------------------------------------------
e=1.6021776e-19 #elementry charge (C or equivalently A.s)
kb=1.380648e-23 #Boltzmann (J/K)
kb_eV=8.6173324e-5 #Boltzmann (eV/K)
angtocm=1e-8 #Angstrom to cm.

c = 2.9979245899e10 # speed of light in vacuum in [cm/s], from Wikipedia.
#kB = 0.6950347      # Boltzman constant in [cm^-1/K], from Wikipedia.
#h_bar = 6.283185    # Reduced Planck constant in atomic unit, where h == 2*pi
#beta = 1.0/(kB * T) #         


def compCond(aType,tdc,N,T=298.15,Hr=1.0,ifPrint=False):
    #Computes conductivity using Nernst-Einstein equation
    sigma=0.0
    for aT in aType:
        if aT in Z: 
            sigma+=N*e**2*Z[aT]**2*tdc/(V*kb*T*Hr*angtocm**3) # should include Faraday constant^2 and gas constant?? see Deng et al 2016
        else: 
            #sigma=0.0
            print("Conductivity can not be determined for all selected atom types.")
            return 0.0
    if ifPrint: print("Conductivity at %d K, sigma=%.5e S/cm."%(T,sigma))
    return sigma

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


#Taken from MDtraj.geometry.rdf.  #http://mdtraj.org/latest/_modules/mdtraj/geometry/rdf.html
#There is also an ASAP version of ASE including RDF.
def compute_rdf(dists,cellVol,r_range=None, bin_width=0.005, n_bins=None, norm_by_V=True):
    """Compute radial distribution functions for interatomic distances along a given trajectory. Adapted from mdtraj pacakge. Returns, bin centre points, g(r) and the coordination(r)

    Parameters
    ----------
    dists: list or array, shape (1,N)
    r_range : array-like, shape=(2,), optional, default=(min(dists), max(dists))
        Minimum and maximum radii.
    bin_width : float, optional, default=0.005
        Width of the bins in nanometers.
    n_bins : int, optional, default=None
        The number of bins. If specified, this will override the `bin_width`
         parameter.

    Returns
    -------
    r : np.ndarray, shape=(np.diff(r_range) / bin_width - 1), dtype=float
        Radii values corresponding to the centers of the bins.
    g_r : np.ndarray, shape=(np.diff(r_range) / bin_width - 1), dtype=float
        Radial distribution function values at r.
    coord : np.ndarray, shape=(np.diff(r_range) / bin_width - 1), dtype=float
        coordination number at r.
    """

    dists=np.array(dists)

    if r_range is None:
        r_range = np.array([min(dists), max(dists)])
    #r_range = ensure_type(r_range, dtype=np.float64, ndim=1, name='r_range',
    #                      shape=(2,), warn_on_cast=False)  #This is a mdtraj property as wel !!
    if n_bins is not None:
        n_bins = int(n_bins)
        if n_bins <= 0:
            raise ValueError('n_bins must be a positive integer')
        bins = np.linspace(r_range[0], r_range[1], n_bins)
    else:
        bins = np.arange(r_range[0], r_range[1] + bin_width, bin_width)

    #distances = compute_distances(traj, pairs, periodic=periodic, opt=opt) #Replace this with ASE !!
    g_r, edges = np.histogram(dists, bins=bins)
    r = 0.5 * (edges[1:] + edges[:-1])


    if norm_by_V:
    # Normalize by volume of the spherical shell.
    # See discussion https://github.com/mdtraj/mdtraj/pull/724. There might be
    # a less biased way to accomplish this. The conclusion was that this could
    # be interesting to try, but is likely not hugely consequential. This method
    # of doing the calculations matches the implementation in other packages like
    # AmberTools' cpptraj and gromacs g_rdf. 
    # VMD and LAMMPS also seem to do this !!!
        V = (4 / 3) * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
        #norm = len(pairs) * np.sum(1.0 / traj.unitcell_volumes) * V
        norm = len(dists) * np.sum(1.0 / cellVol ) * V  #or per volume?
        g_r = g_r.astype(np.float64) / norm  # From int64.
    else:
        norm = len(dists)  #original
        #norm = len(dists) * np.sum(1.0 / cellVol ) 
        g_r = g_r.astype(np.float64) / norm  # From int64.

    coord=np.array([np.sum(g_r[0:i]) for i in range(len(g_r))])

    return r, g_r , coord

def compute_rdf_ASE(dists,cellVol,r_range=None, bin_width=0.005, n_bins=None, norm_by_V=True):

    #From ASE neighbourlist
    d = neighbor_list('d', a, 10.00)
    h, bin_edges = np.histogram(d, bins=100)
    pdf = h/(4*np.pi/3*(bin_edges[1:]**3 - bin_edges[:-1]**3)) * a.get_volume()/len(a)



def Popen4(cmd):
    proc=Popen(cmd,shell=True, stdin=PIPE, stdout=PIPE, close_fds=True)
    out,err=proc.communicate()
    #out=proc.stdout.readline() #does not work.

    if version_info[0] < 3: out2=[x for x in str(out).replace("b'",'').replace("'",'').split("\n") if x!=""] #python3 adds \\n to the end, python2 adds \n
    else:out2=[x for x in str(out).replace("b'",'').replace("'",'').split("\\n") if x!=""]

    return out2,err


def grep(key,fname,n=-1):
    #Uses grep to get the nth line with the keyword from a given file. By default the last occurence is returned.
    try:
        #return popen4('grep -m %d "%s" %s '%(n,key,fname),"r")[1].readlines()[-1][0:-1] #don't take \n at the end.  #Fastest one!!! #Not available in python3
        return Popen4('grep -m %d "%s" %s '%(n,key,fname))[0]
    except:
        return ""


#@jit(nopython=False, nogil=False,parallel=False)  #requires numba library.
def dist(arr1,arr2):
    #Distance btw two atoms. Works with two 1x3 arrays.
    if len(arr1)==4: arr1=arr1[1:4]
    if len(arr2)==4: arr2=arr2[1:4]
    return math.sqrt((arr1[0]-arr2[0])**2+(arr1[1]-arr2[1])**2+(arr1[2]-arr2[2])**2)

def minDist_ASE(arr1,arr2,latt): #needs np.array inputs
    if len(arr1)==4: arr1=arr1[1:4]
    if len(arr2)==4: arr2=arr2[1:4]
    D,x=find_mic(np.array([arr1-arr2]),cell=latt,pbc=True) 
    return x[0]

#@jit(nopython=True, nogil=False,parallel=False)
def minDist(arr1,arr2,latt):
    #Finds the minimum distance from the images applying the PBC. (Minimum image convention).
    #This function works fine for ortho and non-ortho cells, however the find_mic() from ase.geometry is upto 2x faster (using a different algorithm).
    if len(arr1)==4: arr1=arr1[1:4]
    if len(arr2)==4: arr2=arr2[1:4]
    #print arr1,arr2
    minD=dist(arr1,arr2)
    #newarr2=deepcopy(arr2)
    newarr2=[0.0, 0.0, 0.0]
    for i in range(-1,2,1):
        for j in range(-1,2,1):
            for k in range(-1,2,1):
                #newarr2=deepcopy(arr2)
                newarr2=[0.0, 0.0, 0.0]
                newarr2[0]=arr2[0]+i*latt[0][0]+j*latt[1][0]+k*latt[2][0]
                newarr2[1]=arr2[1]+i*latt[0][1]+j*latt[1][1]+k*latt[2][1]
                newarr2[2]=arr2[2]+i*latt[0][2]+j*latt[1][2]+k*latt[2][2]
                currD=dist(arr1,newarr2)
                if currD<minD: minD=currD ;#print minD
    return minD

def RMS(arr1):
    rms=0.0
    for i in arr1: rms+=i**2
    return math.sqrt(rms/len(arr1))


def minRMSD(arr1,arr2,latt): #Not working correctly, input arrays in diff format
    #Finds the minimum RMSD from the images applying the PBC.
    minRMSD=RMSD(arr1,arr2)
    newarr2=dc(arr2)  #don't use deepcopy, takes much longer. (see minDist instead).
    for i in range(-1,2,1):
        for j in range(-1,2,1):
            for k in range(-1,2,1):
                newarr2=dc(arr2)
                newarr2[0]+=i*latt[0][0]+j*latt[1][0]+k*latt[2][0]
                newarr2[1]+=i*latt[0][1]+j*latt[1][1]+k*latt[2][1]
                newarr2[2]+=i*latt[0][2]+j*latt[1][2]+k*latt[2][2]
                currRMSD=RMSD(arr1,newarr2)
                if currRMSD<minRMSD: minRMSD=currRMSD ;#print minD
    return minRMSD

def RMSD(arr1,arr2): 
    #This is a function for calculating the root-mean-square deviation between two sets of values. This is a general fnc, two sets can be either Cartesian coords or internal coords or anything of the same size and no of elements.
    #Only accepts 1-D arrays.
    if len(arr1) == 0 or len(arr2) == 0 or len(arr1) != len(arr2):
        print("RMSD: Sizes of the input arrays don't match. Will return 0.")
        return 0.0

    summ=0.0
    for i in range(len(arr1)):
        for j in range(len(arr1[i])):
            try:float(arr1[i][j])
            except:continue
            summ += (arr1[i][j]-arr2[i][j])**2

    RMSD=math.sqrt(summ/len(arr1))
    return RMSD

def getRMS(coord,alist,latt):
    bonds={}
    data={}

    #for at in alist:
    for i in range(len(coord)):
        #arr1=coord[at][1:4]
        #for i in range(len(coord)):
        for at in alist:
            arr1=coord[at][1:4]
            if i == at: continue #do not count atom's own distance (i.e. 0).
            key=coord[at][0]+'-'+coord[i][0]
            key2=coord[i][0]+'-'+coord[at][0]
            arr2=coord[i][1:4]
            #print arr2

            if key in bonds: bonds[key].append(minDist(arr1,arr2,latt))
            elif key2 in bonds: bonds[key2].append(minDist(arr1,arr2,latt))
            else: bonds[key]=[minDist(arr1,arr2,latt)]

        #print bonds

        keys=list(bonds.keys())
        #keys.sort()
        for ky in keys:  data[ky]=RMS(bonds[ky])
    return data #holds the RMS values of a given bond type

def mean(arr):
  #Returns the mean value of an array.
  mean=0.0
  for ar in arr:
    mean += ar
  mean /= len(arr)
  
  return mean
  
def stddev(arr):
  #Returns the standard deviation of an array.
  mu=mean(arr)
  stddev=0.0
  for ar in arr:
    stddev += (ar-mu)**2
  
  stddev = math.sqrt(stddev/(len(arr)-1))
  return stddev

def vol(latt):#Computes the cell volume from given lattice info
    prod=1.0
    for i in latt:
        summ=0.0
        for j in i:
            summ+=j**2
        prod*=math.sqrt(summ)
    return prod

def volume(cell): #gives the same results as vol(). Checked !!
        return np.abs(np.dot(cell[2], np.cross(cell[0], cell[1])))

def lenCell(cell):
    return [np.linalg.norm(v) for v in cell]

def lenAngleCell(cell,radians=False):#taken from ase.geometry.cell_to_cellpar
    lengths = [np.linalg.norm(v) for v in cell]
    angles = []
    for i in range(3):
        j = i - 1
        k = i - 2
        ll = lengths[j] * lengths[k]
        if ll > 1e-16:
            x = np.dot(cell[j], cell[k]) / ll
            angle = 180.0 / pi * arccos(x)
        else:
            angle = 90.0
        angles.append(angle)
    if radians:
        angles = [angle * pi / 180 for angle in angles]
    return np.array(lengths + angles)


def lenCell_old(cell): #This is essentially a norm operation.
    res=[]
    for i in range(len(cell)):
        summ=0.0
        for j in range(3):
            summ += cell[i][j]**2
        res.append(math.sqrt(summ))
    return res

def compCells(cell1,cell2):#Compare cells.
    len1=lenCell(cell1);    len2=lenCell(cell2)
    return tuple([int(round(len1[i]/len2[i])) for i in range(3)])

def cart2fract(cart_coord):#converts the cartesian coordinates to fractional coordinates using the cell/lattice info.

    a2r = numpy.pi / 180.0

    # get the symmetry information
    a,b,c,alpha,beta,gamma=lenAngleCell(cell,radians=False) #can also be taken as input to reduce comp. cost.

    # convert to radians
    alpha = a2r * alpha
    beta  = a2r * beta
    gamma = a2r * gamma
    
    # (scaled) volume of the cell  
    v = sqrt(1 -cos(alpha)*cos(alpha) - cos(beta)*cos(beta) - cos(gamma)*cos(gamma) + 2*cos(alpha)*cos(beta)*cos(gamma))

    tmat = np.matrix( [
      [ 1.0 / a, -cos(gamma)/(a*sin(gamma)), (cos(alpha)*cos(gamma)-cos(beta)) / (a*v*sin(gamma))  ],
      [ 0.0,     1.0 / (b*sin(gamma)),         (cos(beta) *cos(gamma)-cos(alpha))/ (b*v*sin(gamma))  ],
      [ 0.0,     0.0,                        sin(gamma) / (c*v)                                    ] ]
      )

    fract = cart_coord * tmat.T

    # return the Nx3 results
    return fract


#############
# Functions for MSD #
#############

#@jit(nopython=True, nogil=False,parallel=True)
#@njit(parallel=True)
def getJumps_serial(atoms_list,t0,noSteps,closestSites,lattice,coords,pminSite=None,pwyckoff=None,verb=0):
    trackJumps=[ 0 for i in range(len(atoms_list)) ]
    trackSites={}

    #if verb:print "Progress (total=%d): "%len(atoms_list),;stdout.flush()
    for i in range(len(atoms_list)): #loop over atoms
        at=atoms_list[i]
        #if verb:print at, ; stdout.flush()
        pminSite=None
        for st in range(t0,noSteps): #loop over time steps.
            md=float(10**8)
            minSite=[]
            for j in range(len(closestSites[i][0])): #loop over crys sites.i.e. NNB or nN.
                site2=closestSites[i][0][j][0] #orig: [at] #this does not work when atoms_list does not start from 0.
                cwyck=closestSites[i][2][j] #orig: [at] 
                #x=minDist(coords[st][at],site2,lattice)
                x=minDist_ASE(coords[st][at],site2,lattice) #~2x faster than minDist, same results.
                #x=dist(coords[st][at],site2) 

                #if x < md: minSite=deepcopy(site2);md=x
                if x < md: minSite=site2;md=x;cwyckoff=cwyck

            if pminSite==None: pminSite=minSite;pwyckoff=cwyckoff;#continue #first step
            elif pminSite[0] != minSite[0] or pminSite[1] != minSite[1] or pminSite[2] != minSite[2]:
                trackJumps[i] += 1
                pminSite=minSite

                wkey=pwyckoff+'-'+cwyckoff
                rwkey=cwyckoff+'-'+pwyckoff
                if wkey in trackSites:                trackSites[wkey]+=1
                elif rwkey in trackSites:                trackSites[rwkey]+=1
                else: trackSites[wkey]=1

                pwyckoff=cwyckoff

    return trackJumps,trackSites

def getJumps_coords(atoms_list,t0,chunk,closestSites,lattice,coords,nprocs=None,verb=0):
        #parallelise over time steps (coords).
        pcoords=[]
        for part in range(t0,len(coords)-chunk+1,chunk): 
            if part==0:pcoords.append(coords[part:part+chunk]) #orig start from part
            else: pcoords.append(coords[part-1:part+chunk]) #not to miss the jumps from the final step of the previous junk.
        try:pcoords[-1].extend(coords[part+chunk:]) #make sure jumps at the final step(s) are included.
        except:None

        with concurrent.futures.ProcessPoolExecutor(max_workers=nprocs) as executor:
            future_to_atom = {executor.submit(getJumps_serial,atoms_list,0,len(pcrd),closestSites,lattice,pcrd,verb=verb):pcrd for pcrd in pcoords}
            cnt=0
            if verb: print("Progress (total=%d): "%len(pcoords), )
            if 0: #get the results on the fly.
                for future in concurrent.futures.as_completed(future_to_atom):
                    part = future_to_atom[future]
                    try:
                        data = future.result()
                    except Exception as exc:
                        print(('%r generated an exception: %s' % (cnt, exc)))
                    else:
                        cnt+=1
                        #print(atom,data)
                        if args.verb:
                            print("Part: ", cnt, )
                            stdout.flush()
                            print(data[0],data[1])
                        for atom in range(len(atoms_list)):
                            trackJumps[atom]+=data[0][atom] 
                        tSites=data[1]
                        for skey in list(tSites.keys()):
                            if skey in trackSites:
                                trackSites[skey]+=tSites[skey]
                            else:
                                trackSites[skey]=tSites[skey]
            else: # wait for all processes to finish (or fail). No diff in comp. times.
                done,not_done=concurrent.futures.wait(future_to_atom) #wait all processes to finish.Returns a named 2-tuple of sets: done and not_done (unsuccessful).
                if len(not_done)>0:print("Unsuccesful processes: ",not_done)
                for future in done:
                    cnt+=1
                    atom = future_to_atom[future]
                    data = future.result()
                    if verb:print("Part: ", cnt, data[0],data[1])
                    for atom in range(len(atoms_list)):
                            trackJumps[atom]+=data[0][atom] 
                    tSites=data[1]
                    for skey in list(tSites.keys()):
                        if skey in trackSites:
                            trackSites[skey]+=tSites[skey]
                        else:
                            trackSites[skey]=tSites[skey]
                            #print trackSites
                #print

        return trackJumps,trackSites

def getJumps_atoms(atoms_list,t0,chunk,closestSites,lattice,pcoords,nprocs=None,verb=0):
#This function parallelise the jump counting over the selected atoms.
    trackSites={}#to hold the number of jumps between diverse crys. sites.
    trackJumps=[ 0 for i in range(len(atoms_list)) ]
    #if nprocs != 1: #parallelise over atoms.
    with concurrent.futures.ProcessPoolExecutor(max_workers=nprocs) as executor:
        future_to_atom = {executor.submit(getJumps_at,i,atoms_list[i],t0,chunk,closestSites,lattice,pcoords): i for i in range(len((atoms_list)))}
        cnt=0
        if verb: print("Progress (total=%d): "%len(atoms_list), )
        if 1: #get the results on the fly.
            for future in concurrent.futures.as_completed(future_to_atom):
                atom = future_to_atom[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print(('%r generated an exception: %s' % (atom, exc)))
                else:
                    cnt+=1
                    if verb:  print("Part : ",cnt, data[0],data[1])
                    stdout.flush()
                    trackJumps[atom]+=data[0]
                    tSites=data[1]
                    for skey in list(tSites.keys()):
                        if skey in trackSites:
                            trackSites[skey]+=tSites[skey]
                        else:
                            trackSites[skey]=tSites[skey]
        else: # wait for all processes to finish (or fail). No diff in comp. times.
            done,not_done=concurrent.futures.wait(future_to_atom) #wait all processes to finish.Returns a named 2-tuple of sets: done and not_done (unsuccessful).
            if len(not_done)>0:print("Unsuccesful processes: ",not_done)
            for future in done:
                cnt +=1
                atom = future_to_atom[future]
                data = future.result()
                #print data
                if verb:  print("Part : ",cnt, data[0],data[1])
                trackJumps[atom]+=data[0]
                tSites=data[1]
                for skey in list(tSites.keys()):
                    if skey in trackSites:
                        trackSites[skey]+=tSites[skey]
                    else:
                        trackSites[skey]=tSites[skey]
    return trackJumps,trackSites

def getJumps_at(ind,at,t0,noSteps,closestSites,lattice,coords,pminSite=None,pwyckoff=None):
#This function is for counting number of jumps for a given atom over a given time chunk. 
#Called in parallel by getjumps_atoms().
    trackJumps=0 #tracks no of jumps of each atom in the list
    trackSites={} #tracks the no of jumps btw given crys. sites.
    pminSite=None

    #closestSites[atom_list][([crys_sites][distances][wyckoffs]) x nN in increasing order of distance]

    for st in range(t0,noSteps): #loop over time steps.
        md=float(10**8)
        minSite=[]
        for j in range(len(closestSites[ind][0])): #loop over crys sites. 
            site2=closestSites[ind][0][j][0]
            cwyck=closestSites[ind][2][j] #original
            #x=minDist(coords[st][at],site2,lattice)
            x=minDist_ASE(coords[st][at],site2,lattice) #use this with wrapped coords.
            #x=dist(coords[st][at],site2) #gives the same results as other two but 10x faster.

            if x < md: minSite=site2;md=x;cwyckoff=cwyck

        if pminSite==None: pminSite=minSite;pwyckoff=cwyckoff;continue #first step
        elif pminSite[0] != minSite[0] or pminSite[1] != minSite[1] or pminSite[2] != minSite[2]:
            trackJumps += 1           

            pminSite=minSite
            wkey=pwyckoff+'-'+cwyckoff
            rwkey=cwyckoff+'-'+pwyckoff
            if wkey in trackSites:                trackSites[wkey]+=1
            elif rwkey in trackSites:                trackSites[rwkey]+=1
            else: trackSites[wkey]=1

            pwyckoff=cwyckoff

    return trackJumps,trackSites

def combineJumps(trackSites):
    #Combine same jump types in reverse order (only occurs when using multiprocesses.)
    keys=list(trackSites.keys())
    trackSites2={}
    used=[]
    for i in range(len(keys)):
        if i in used: continue
        k1=keys[i]
        k1s=k1.split("-")
        trackSites2[k1]=trackSites[k1]
        for j in range(i,len(keys)):
            k2=keys[j]
            k2s=k2.split("-")
            if (k2s[0]==k1s[1] and k2s[1]==k1s[0]) and k1 != k2 : 
                used.append(j);

                if 1: # add up the back-and-forth transitions (as also done by Wagemaker et al.)
                    trackSites2[k1]=trackSites[k1]+trackSites[k2];
                else:
                    diff=trackSites[k1]-trackSites[k2];#subtraction as returns to original position??
                    if diff>0:trackSites2[k1]=diff
                    else: trackSites2[k2]=abs(diff)

    return trackSites2

def getAvgCrysSiteDist_old(cSites,lattice):
    avg=0.0 #overall avg. distance btw crys. sites.
    cnt=0
    dists={};cnts={}#dict holding the avg dist for each site pair
    for i in range(len(cSites)-1):
        for j in range(i+1,len(cSites)):
            key1="%s-%s"%(cSites[i][1],cSites[j][1])
            key2="%s-%s"%(cSites[j][1],cSites[i][1])
            x= minDist(cSites[i][0],cSites[j][0],lattice)
            #avg+=x
            if key1 in dists or key2 in dists:
                dists[key1]+=x; cnts[key1]+=1
                if key2!=key1:  dists[key2]+=x;cnts[key2]+=1
            else:
                dists[key1]=x;  dists[key2]=x
                cnts[key1]=1; cnts[key2]=1
            cnt+=1
    #avg /= cnt
    for key in dists:
	    dists[key]/=cnts[key]

    avg=np.mean(list(dists.values()))
    #return distance dict for site pairs and the overall average.,lattice
    return dists,avg 

def getAvgCrysSiteDist(cSites,lattice,r_range=None,n_bins=None,bin_width=0.005):
    avg=0.0 #overall avg. distance btw crys. sites.
    cnt=0
    dists={};cnts={}#dict holding the avg dist for each site pair
    min_dists={}
    for i in range(len(cSites)-1):
        for j in range(i+1,len(cSites)):
            key1="%s-%s"%(cSites[i][1],cSites[j][1])
            key2="%s-%s"%(cSites[j][1],cSites[i][1])
            x= minDist_ASE(cSites[i][0],cSites[j][0],lattice) #faster than minDist
            #avg+=x
            if key1 in dists or key2 in dists:
                dists[key1].append(x); cnts[key1]+=1
                if key2!=key1:  dists[key2].append(x);cnts[key2]+=1
            else:
                dists[key1]=[x];  dists[key2]=[x]
                cnts[key1]=1; cnts[key2]=1
            cnt+=1
    #avg /= cnt
    if r_range is None:
        r_range = np.array([0.0, 1.0])
    #r_range = ensure_type(r_range, dtype=np.float64, ndim=1, name='r_range', shape=(2,), warn_on_cast=False)
    if n_bins is not None:
        n_bins = int(n_bins)
        if n_bins <= 0:
            raise ValueError('n_bins must be a positive integer')
        bins = np.linspace(r_range[0], r_range[1], n_bins)    
    else:
        bins = np.arange(r_range[0], r_range[1] + bin_width, bin_width)

    for key in dists:
	#dists[key]/=cnts[key]
        hist, edges = np.histogram(dists[key])#, bins=bins)
        #print key, hist, edges
        size=edges[1]-edges[0]
        min_dists[key]=np.mean([ds for ds in dists[key] if ds <= (edges[0]+edges[1])/2])


    avg=np.mean(list(min_dists.values()))
    #return distance dict for site pairs and the overall average.,lattice
    return min_dists,avg 

#############
# General Functions  #
#############
def getLattice(inpf): #returns the lattice information from an input xyz file.
    inpf.seek(0)
    #We don't want to store all lines in memory.
    lines=[]
    for i in range(2):#just  two lines.
        lines.append(inpf.readline())
    inpf.seek(0)    #print inpf.tell()

    #We can get the lattice infpo from POSCAR file using ase.io !!
    try: noAtoms=int(lines[0])
    except: print("No number of atoms info in input .xyz file"); exit()
    try: 
        lattice=[]
        x=lines[1].split('"')[1].split()
        #print x
        z=[]
        for j in range(len(x)):
            z.append(float(x[j]))
            if (j+1)%3==0: lattice.append(z);z=[]      

        return lattice
    except: print("No lattice info in the input file."); exit()


def getCoords(inpf): #Should we skip the discarded steps in reading directly (to speed-up??)
    inpf.seek(0)
    coords=[] #This holds the atomic coords along the trajectory given in OUTCAR.xyz.
    cstep=[]
    noSteps=0
    #noAtoms=int(lines[0])
    for line in inpf: #this reduces memory requirements. One can also use inpf.xreadlines(), but this one is newer.
        line=line[0:-1]
        #print line
        if len(line)==0: continue
        try:
            words=line.split()
        #if re.search("Lattice",words[0]):
            if "Lattice" in words[0]:
                if noSteps!=0: 
                    coords.append(cstep); cstep=[]
                    #if len(coords[-1])!=noAtoms: print "WArning: step %d does not have coordinates for all atoms."%noSteps
                noSteps += 1
            elif len(words)==4:
                cstep.append([words[0],float(words[1]),float(words[2]),float(words[3])])
            #print cstep
            elif len(words)<=1: continue
            #if line==lines[-1]: coords.append(cstep);#noSteps += 1
        except:
            print("getCoords: Error in line \n",line) 

    coords.append(cstep) #manually add the last step.
    #print len(coords),noSteps
    if noSteps!=len(coords): print('Problem with the number of steps read  in OUTCAR.xyz.',noSteps,len(coords)); exit()
    else: print("%d time steps were read from input."%noSteps)
    #coords=array(coords)
    lattice=getLattice(inpf)

    return [coords,noSteps,lattice]


def getCoords_ASE(inpf,t0=0,tf=1e10,fract=False,unwrap=False,write=False,skip=1,wrap=False):
    #unwrap option is for unwrapping the wrapped input coordinates and wrap is vice versa. Unwrapped coordinates are needed for MSD and VACF analyses, but mean jump rate (MJR) needs wrapped coordiantes as input.
    #tsteps=[tr for tr in ase.io.iread(inpf)]
    tsteps=[]
    cnt=0
    for i, tr in enumerate(ase.io.iread(inpf)):
        cnt+=1
        if (t0<=i<tf) and  i % skip==0: tsteps.append(tr)
    

    init_pos=tsteps[0].get_scaled_positions()
    ppos=dc(init_pos)
    noAtoms=len(tsteps[0])
    shifts=np.zeros((noAtoms,3))

    coords=[];coords_arr=[]
    if unwrap:  print("Unwrapping the coordinates as read from %s."%inpf)
    for st in range(0,len(tsteps)):#,skip):
        atoms=tsteps[st]
        if unwrap and st!=0:
            cpos=atoms.get_scaled_positions(wrap=False)
            #shifts=np.zeros((noAtoms,3))
            for at in range(noAtoms):
                for j in range(3):
                    diff=cpos[at][j]-ppos[at][j]
                    if np.fabs(diff) >0.5: #also covers the >1.0 differences.
                        shifts[at][j]+=(diff - np.sign(diff))
                    else: shifts[at][j]+=diff 

            tsteps[st].set_scaled_positions(init_pos+shifts)
            ppos=dc(cpos)

        tmp=[]
        if fract: pos=tsteps[st].get_scaled_positions(wrap=wrap)
        else:  pos=tsteps[st].get_positions(wrap=wrap) 

        coords_arr.append(pos) #Array of Nsteps x Natoms x 3

        """
        for i in range(len(atoms)): #Delete this and add aTypes variables and modify the other fncs to use this rather than coords.
            tmp.append([atoms[i].symbol, pos[i][0],pos[i][1],pos[i][2]])
        coords.append(tmp)
        """
    noSteps=len(coords_arr)
    lattice=tsteps[0].get_cell()
    #stoich=tsteps[0].get_chemical_formula(mode='hill')
    stoich,fu=getReduced(tsteps[0])
    aTypes=tsteps[0].get_chemical_symbols()

    #if unwrap and write:
    if write: #Fix this: ite only skipped steps.
        print("Writing current atomic positions in saved.xyz...")
        
        if 0:ase.io.write("saved.xyz",tsteps,format="xyz",append=0) #writes only the unwrapped ones by default.
        else:writeCoords("saved.xyz",coords_arr,aTypes,lattice)

    print("%d timesteps were found in %s, %d timesteps will be used in the analysis."%(cnt,inpf,noSteps))


    #return coords,noSteps,array(coords_arr),lattice
    return coords,array(coords_arr),lattice,stoich,fu,aTypes

def writeCoords(outf,coords,aTypes,lattice): #Write coordinates in an xyz file (can be interfacewd with ASE for other file types.#coords is in coords[ts][atomid][aType,x,y,z] format.
    outf=open(outf,"w")
    noSteps=len(coords)
    noAtoms=len(coords[0])
    str2=""
    for lt in lattice: 
        for l in lt:
            str2+="%9.6f "%float(l)
    str1="%d\n"%noAtoms
    str1+='Lattice="%s" \n'%str2[0:-1]
    for st in range(noSteps):
        outf.write(str1)
        for at in range(noAtoms):
            crd=coords[st][at]
            outf.write("%3s %9.6f %9.6f %9.6f\n"%(aTypes[at],crd[0],crd[1],crd[2]))
        #outf.write("\n")
    outf.close()

def getReduced(atoms):
    #Determine the formula unit and reduced formula.
    if isinstance(atoms,str):
        atoms=Atoms(atoms)
    elif isinstance(atoms,ase.Atoms):
        atoms=atoms

    ats={}
    aTypes=atoms.get_chemical_symbols()
    for at in aTypes:
        if at in ats: ats[at]+=1
        else: ats[at]=1

    #ats=re.findall(r'([A-Z][a-z]*)(\d*)', stoich) #gives [('H', '2'), ('S', ''), ('O', '4')]

    try:
        fu=1
        vals=list(ats.values())
        for i in range(2,min(vals)+1):
            fl=1
            for j in vals:
                if j%i!=0:fl=0
            if fl:fu=i
    except: print ("Error in fu determination, fu set to 1");   fu=1

    reduced=""
    keys=sorted(ats.keys())
    for key in keys:
        reduced += key
        if ats[key]/fu!=1:reduced+=str(int(ats[key]/fu)) 

    return reduced,fu


#############
#Functions for VAFC #
#############
def read_data(filename, sel):
    with open(filename, 'r') as fo:
        timestep = 0
        coords = []
        
        for line in fo:
            try:
                next(fo)
            except StopIteration:
                break
            if isinstance(sel, int) and sel == Natom:
                for n in range(sel):
                    line = next(fo)
                    info = line.split()
                    coords.append(info[1:])
                timestep += 1
        
            elif isinstance(sel, list) and len(sel) >= 2:
                for k in range(max(sel) + 1):
                    line = next(fo)
                    info = line.split()
                    coords.append(info[1:])
                timestep += 1
                try:
                    for k in range(Natom - max(sel) - 1):
                        next(fo)
                except StopIteration:
                    break
        coords = np.asfarray(coords, dtype=np.float64).reshape(timestep,-1,3)
    #if isinstance(sel, list) and len(sel) == 1 and sel[0] == Natom:
    if isinstance(sel, int) and sel == Natom:
        return coords
    elif isinstance(sel, list) and len(sel) >= 2:
        return coords[:,np.array(sel),:]
    

def calc_derivative(array_1D, delta_t,axis=-1):
    ''' The derivatives of the input 1D array were obtained by using the
    finite differences method.
    '''
    dy = np.gradient(array_1D,axis=axis)
    return np.divide(dy, delta_t)

def choose_window(noSteps, typ='Gaussian'):
    if typ == 'Gaussian':
        sigma = 2 * math.sqrt(2 * math.log(2))
        #std = float(raw_input(standard)) 
        std=2000
        window_function = signal.gaussian(noSteps, std/sigma, sym=False)

    elif typ == 'BH':
        window_function = signal.blackmanharris(noSteps, sym=False)
    
    elif typ == 'Hamming':
        window_function = signal.hamming(noSteps, sym=False)
    
    elif typ == 'Hann':
        window_function = signal.hann(noSteps, sym=False)
    
    return window_function

def calc_FFT(array_1D, window):
    '''
    This function is for calculating the "intensity" of the ACF at each frequency
    by using the discrete fast Fourier transform.
    '''
####
#### http://stackoverflow.com/questions/20165193/fft-normalization
####
    WE = sum(window) / len(array_1D)
    wf = window / WE
    # convolve the blackman-harris window function.
    
    sig = array_1D * wf
	
    # A series of number of zeros will be padded to the end of the \
    # VACF array before FFT.
    N = zero_padding(sig)

    yfft = np.fft.fft(sig, N, axis=0) / len(sig)
#    yfft = np.fft.fft(data, n=int(N_fft), axis=0)/len(data) # no window func.
    #print "shape of yfft", np.shape(yfft)
    return np.square(np.absolute(yfft))

def zero_padding(sample_data):

    ''' A series of Zeros will be padded to the end of the dipole moment
    array (before FFT performed), in order to obtain a array with the
    length which is the "next power of two" of numbers.
    
    #### Next power of two is calculated as: 2**math.ceil(math.log(x,2))
    #### or Nfft = 2**int(math.log(len(data_array)*2-1, 2))
    '''
    return int(2 ** math.ceil(math.log(len(sample_data), 2)))



def calc_ACF(array_1D):
    # Normalization
    yunbiased = array_1D - np.mean(array_1D, axis=0)
    ynorm = np.sum(np.power(yunbiased,2), axis=0)
    
    autocor = signal.fftconvolve(array_1D,
                                 array_1D[::-1],
                                 mode='full')[len(array_1D)-1:] / ynorm
    return autocor

#############
#Main script#
#############
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computes the RMS of different bond types for a given list of atoms along a MD trajectory.')

    #General options
    parser.add_argument('-v','--verb',default=False,action='store_true',help='Verbose output.')

    parser.add_argument('-i', '--inpf', type=str,default="all-traj.xyz",
                        help='Input .xyz file. Default: all-traj.xyz that is extracted from any code with lattice information (similar to ASE format).')
    parser.add_argument('-n', '--atoms_list', nargs='*', type=int,#required=True
                        help='list of atoms for whom the nearest neighbors should be computed (count starts at 1). Example usage: -n 1 3 4 ')
    parser.add_argument('-nl', '--list',
                        help='atom list(count starts at 1); e.g. -nl 1:128')
    parser.add_argument('-at', '--aType', type=str,nargs='*', 
                        help='atom type to be included ; e.g. -at Na; -at all or -at a for all types')
    parser.add_argument('-t', '--type',choices=["bond","dist","jump","vacf","rdf",'vanHove','tort',"a"],required=True, help='Determines the type of analysis. Default: all (i.e -t a). Other options are: \n b: Runs a root-mean-square (RMS) analysis on bond length for each bond type assoicated with the selected atoms. \n d: switches on the calculation of the mean-square diffusivity (MSD, in cm2/s) for the selected list of atoms and the resulting conductivity (S/cm)')

    parser.add_argument('-t0', '--t0',type=int, default=0, help='To set the initial time step different than 0. Useful to discard the equilibration part of a MD trajectory. ')
    parser.add_argument('-tf', '--tfin',type=int,  default=None, help='To set the final time step to truncate the trajectory. One could set -tf 1 for analysing only the first frame. Def: all time steps after args.t0 are considered ')

    parser.add_argument('-skip', '--skipSteps',type=int, default=5, help='To skip some time steps in the MD trajectory. Def: all frames are considered. ')

    parser.add_argument('-T', '--T',type=float, default=0.0, help='To enter the MD temperature [K] based on which the conductivity for the selected atoms will be computed. Default: taken from the INCAR file.')

    parser.add_argument('-ct','--corrTemp',default=False,action='store_true',
                        help='This option is for computing the corrected temperature information (using (T*$nions)/($nions-1)), as default VASP output is not correct.')

    parser.add_argument('-Hr', '--HavenRatio',type=float, default=1.0, help='Haven ratio between conductivity and diffusion coefficient. Def: Hr=1.0')

    parser.add_argument('-ts', '--timeStep',type=float, default=2.0, help='Time step in the simulations. Def: 2.0 fs')

    parser.add_argument('-dim', '--dimension',type=int, choices=[1,2,3],default=3, help='Dimension of the ionic migration. Dif: 3D.')

    parser.add_argument('-uw','--unwrap',default=False,action='store_true', help='Generate unwrapped coordinates needed for MSD and VFAC, in case input coordinates are wrapped into cell. Def: No unwrapping done.')

    parser.add_argument('-save','--saveGeom',default=False,action='store_true',help='Save the actual coordinates used in analysis in saved.xyz. This option can be used for saving the unwrapped/wrapped coordinates generated by this script. Def: False')

    parser.add_argument('-noplot','--noplot',default=False,action='store_true',help='Surpress plotting')

    parser.add_argument('-da','--detailed',default=False,action='store_true', help='Detailed analysis')

    #relevant to MJR method.

    parser.add_argument('-ic', '--initCrysFile', type=str,help='File for obtaining the initial crystalline site information. Can be any type that ASE supports.')

    parser.add_argument('-tol', '--tol',type=float, default=1e-1,help="The symmetry tolerance for determining the equivalent sites from the reference crystal (see -ic option). Def: 0.1")

    parser.add_argument('-prim','--ifPrim', default=False,action='store_true', help='Use the primitive cell of the reference crystal instead of supercell (requires SPGlib)')

    parser.add_argument('-as', '--allowedSites',type=str,nargs="*",help="To specify the allowed tranisition sites for migratign ions in the mean-jump rate (MJR) method. E.g. -as 4d 2a 6b 12d. Def: all sites are considered in determining jumps.")

    parser.add_argument('-lj','--limitJumps', default=False,action='store_true', help='Limit the jump analysis to the ones between the crystalline sites occupied by the given atom type (see -at option). This is a sepcial case of the -as option (see above). One could use -as to explicitly state which crys. sites to include in the MJR analysis, too. Def: all srys. sites are considered.')

    parser.add_argument('-nN', '--Nneighbours',type=str, default=8, help='Number of closest neighbours to include in the mean jump rate analysis (e.g. -t j). Warning: Longer calculations when more neighbours considered. Type "-nN all" for considering all sites. Default: 8')

    parser.add_argument('-np', '--nprocs',type=int,default=1, help='To limit the ,aximum number of CPUs to use. Default: #processes is equal to # CPUs available.')

    parser.add_argument('-parAtoms','--parAtoms', default=False,action='store_true', help='Parallelise over atoms (in the selected list) rather than time steps (as default).')


    #relevant to MSD method.
    parser.add_argument('-dncTDC','--discardNonConvergedTDC',default=False,action='store_true', help='To discard the data whereby the tracer diffusicivity coefficient (TDC) is not converged to preset tolerance (see -tdctol). Def: all data is used for MSD analysis.')

    parser.add_argument('-tdctol', '--tdctol',type=float, default=1e-7,help="The TDC convergence tolerance.")

    #relevant to VACF method.
    parser.add_argument('-FFTtype', '--FFT_window_type',choices=['Gaussian',"BH",'Hamming','Hann'],default='Gaussian', required=False, help='The window function type for FFT calculations. BH:Blackman-Harris')
    parser.add_argument('-FFTstd', '--FFTstd',type=float, default=2000,help="Std value for Gaussian-type broadening for FFT. The larger the std, the narrower the Gaussian line-shape. Default: 2000 (which yields roughly 20 cm^1 for a 10k step trajectory).")



    args = parser.parse_args()

    initT=time.time()

    #########
    #Read atomic coordinates from the input file.
    if args.tfin: print("Only considering the first %d time steps as requested."%args.tfin)
    if args.t0: print("Discarding the first %d time steps as requested."%args.t0)

    inpf=open(args.inpf,"r")
    if args.skipSteps <1: args.skipSteps=1
    #if args.type=="vacf":args.skipSteps=3
    if args.type=="rdf" and args.skipSteps <100:
        print("Computational effort is not justified for a finer calculation of RDF. Consider setting the skip steps to 100 or larger (i.e. -skip 100) ...")
        #args.skipSteps=100
    if args.skipSteps !=1 : print("Skipping %d steps..."%args.skipSteps)

    if 0: coords,noSteps,lattice=getCoords(inpf) #Old function, no unwrapping available.
    #if args.type=="dist" or args.type=="a": coords,noSteps=getCoords_ASE(args.inpf,unwrap=1,write=0)
    else:
        if args.type=="jump":   wrap=1 #MJR analysis needs wrapped coords as input.
        elif  args.type=="rdf": wrap=0
        else:             wrap=False #MSD and VACF need unwrapped coords as input.

        if args.tfin: tfin=args.tfin
        else: tfin=1e10
        if tfin==0: tfin=1e10
        
        if args.type not in ["vanHove",'tort']: 
          coords,coords_arr,lattice,stoich,fu,aTypes=getCoords_ASE(args.inpf,unwrap=args.unwrap,write=args.saveGeom,skip=args.skipSteps,wrap=wrap,t0=args.t0,tf=tfin) #t0=args.t0 #but first change args.t0 to 0 in the main code.
          noSteps=len(coords_arr); del coords
          noAtoms=len(aTypes)
        else:
          #coords_arr=[];noSteps=0
          coords_arr=ase.io.read(args.inpf,index='0::%d'%args.skipSteps); aTypes=coords_arr[0].symbols;noSteps=len(coords_arr);noAtoms=len(aTypes);lattice=coords_arr[0].get_cell()
          print ("%d timesteps will be used in the analysis."%noSteps)

    
    if args.verb: 
        a, b, c, alpha, beta, gamma=ase.geometry.cell_to_cellpar(lattice)
        print("Cell parameters: ",a, b, c, alpha, beta, gamma)

    print("Elapsed time: %.2f sec."%( time.time()-initT))



    ##########
    #Get the user input as to atom selection.
    if args.aType: #Determine the atomic indices of the atoms with the desired type.
        if args.aType in ['all','a'] :atoms_list=list(range(noAtoms))
        else:
            atoms_list=[]
            for i in range(noAtoms):
                #at=coords[0][i][0]
                at=aTypes[i]
                if  at in args.aType: atoms_list.append(i)  

        if len(atoms_list)==0: print("Requested atom type (%s) was not found in the input file. Terminating..."%args.aType);exit()
    elif args.atoms_list:
        atoms_list=[] #args.atoms_list
        #atoms_list = np.array(args.atoms_list) - 1
        for at in args.atoms_list: 
            if at == 0: print("Wrong atom index (atom count starts at 1)");exit()
            else: atoms_list.append(at-1) #atoms_list[i] = int(atoms_list[i])-1
    elif args.list:
        x=args.list.split(':')
        x0=int(x[0])
        if len(x[1])==0: x1=noAtoms
        else: x1=int(x[1])
        if x0==0 or x1>noAtoms:
            print("Wrong atom index (atom count starts at 1)");exit()  #atoms_list = range(len(atoms))
        atoms_list=list(range(x0-1,x1))
        #print "atoms_list: ", atoms_list
    else:
        #atoms_list = range(len(atoms))
        print("The list of atoms for which the RMS of bond lengths will be computed must be given. Use -n, -nl or -at for this purpose. See help (-h) for furhter information.")
        exit()

    print("%d atoms selected: %s"%(len(atoms_list),atoms_list ))
       
    ####################
    # Common variables #
    ####################

    #if args.tfin: tfin=args.tfin
    #else:tfin=noSteps
    dt=args.timeStep #fs take it from user or from INCAR.
    dim=args.dimension  #2D/3D dimension.
    N=len(atoms_list) #no of diffusing ions.
    V=vol(lattice) #volume (A^3)
       #T=525 #Kelvin, get it from INCAR file or as input.
    Hr=args.HavenRatio

    if args.verb: print("number density of selected ions: %.3e cm^-3"%(N/V*1e24))

    T=0
    if not args.T:#get it from INCAR.
        flist=os.listdir(".")
        seeds=[x for x in flist if ".param" in x]
        #if os.path.exists('./INCAR'):
      
        #elif os.path.exists('./*.param'):
        if len(seeds)>0: 
            #if len(seeds)==1:seed=seeds[0].split(".param")[0]
            #else: print "More than one .param files found, using %s"%seeds[0];
            seed=seeds[0].split(".param")[0]
            try:
                T=float(popen("grep -i 'md_temperature' %s.param"%seed).readlines()[0][0:-1].split("=")[1].split("K")[0])
                #T=grep('TEBEG=',"INCAR",1).split(";")[0].split("=")[1]
                print("Temperature (%.1f K) was read from %s.param..."%(T,seed))
            except:    T=0

        elif "INCAR" in flist:
            try:
                T=float(popen("grep 'TEBEG=' INCAR").readlines()[0][0:-1].split(";")[0].split("=")[1])
                #T=grep('TEBEG=',"INCAR",1).split(";")[0].split("=")[1]
                print("Temperature (%.1f K) was read from INCAR..."%T)
            except: T=0
          
    else: T=args.T

    if T==0:
        print("Temperature info could not be found in INCAR/param file. Setting T=298K...")
        T=298

    #Correct the temperature if requested.  Check this option !!
    if args.corrTemp:
        print("Creating the corrected temperature info.")
        str1="""#! /bin/bash
    nions="`grep ' ions' vasp.out | awk '{print $7}'`"
        grep F= vasp.out | awk '{print $3}'| awk '{ print ($1*$nions)/($nions-1)}'  > temp.tmp
    """
        os.system(str1)

     ###########################
     ### Part for RMS bond distance analysis ###
     ###########################
    if args.type=="bond" or args.type=="a":
        rms_dict={}
        for st in range(noSteps):
    #for st in range(100):
            x=getRMS(coords[st],atoms_list,lattice)
            keys=list(x.keys())
        #keys.sort()
            for key in keys:
                if key in rms_dict:rms_dict[key].append(x[key])
                else:rms_dict[key]=[x[key]]

        outf=open("rms-bonds-new.dat","w")
        keys=list(rms_dict.keys())
    #print keysargs.
        keys.sort()
        str1="#Step  "
    #vals=[]

        print("These bond types were considered in the analysis:", keys)
        for key in keys: 
            str1+="%-8s"%key
        #vals.append(rms_dict[key].values())
        outf.write(str1+"RMSD  \n")
        #for st in range(args.t0,noSteps):
        for st in range(0,noSteps):
            str1="%-7d"%st
            for key in keys: 
                y=abs(rms_dict[key][st]-rms_dict[key][0])/rms_dict[key][0]
                str1+="%-8.5f"%y
        #print coords[0][1:4]
        #str1+="%-8.5f"%(minRMSD(coords[st],coords[0],lattice))
            str1+="%-8.5f"%(RMSD(coords[st],coords[0]))
            outf.write(str1+"\n")
        outf.close()

        print("Elapsed time: %.2f sec."%( time.time()-initT))

    if args.type=="dist" or args.type=="a": #Compute the diffusivity here.
        #Use atoms in the toms_list for now, then update it to use all atoms of a given atom type.
        MSD=[[],[]] #mean-square-displacement for all migrating ions over time.
        MSD=zeros((2,noSteps))
        #MSD=zeros((2,tfin))
        tdc=[]#tracer diffusion coefficient (for all migrating ions) over all trajectory.
        err=args.tdctol #relative change/error for TDC(D*).

        ifConv=0 #If the tdc has converged.
        convSt=0   
        summ=0.0
        atomMSDs=[0.0 for i in range(len(atoms_list))]
        p_vals=[];         r2_vals=[] ; stderrs=[]#for statistics
        #for st in range(args.t0,noSteps):#,args.skipSteps): #loop over time #check the start point.
        ##for st in range(args.t0,tfin):#,args.skipSteps): #loop over time #check the start point.
        for st in range(noSteps):#,args.skipSteps): #loop over time #check the start point.
            if  1:
                summ=0.0
                for i,at in enumerate(atoms_list): #loop over atoms; this could be used for analysing each individual atom.
                    x=dist(coords_arr[st][at],coords_arr[0][at]) ;   summ += x**2
                    if args.detailed: atomMSDs[i]+=x**2 #add cooperitivity here.
                    #x=dist(coords[st][at],coords[st-1][at]);    summ += x**2

            else: #using numpy built-in function do not bring over much of a speed-up (< 5%).
                summ=sum(np.linalg.norm(coords_arr[st][atoms_list]-coords_arr[0][atoms_list],axis=1)**2) #only consider atoms in the atoms_list.
                #summ+=sum(np.linalg.norm(coords_arr[st][atoms_list]-coords_arr[st-1][atoms_list],axis=1)**2) #only consider atoms in the atoms_list.

            #if args.detailed: atomMSDs+=summ

            #Correct the time step (no need)
            #st -= args.t0

            MSD[0][st]+=(dt*(st+args.t0+1)*args.skipSteps)/1000 #fs -> ps for convenience.
            MSD[1][st]+=summ/N

            #if st==args.t0: tdc.append(0.0);continue
            if st==0: tdc.append(0.0);continue

            exclZero=0
            if exclZero:    #excluding the first point (i.e. time step) in MSD as there might be a huge jump.
                x=(MSD[0][1:])*1000 #simulation time (back to fs)
                y=MSD[1][1:] #MSD(D*)
                
            else:
                x=(MSD[0][0:])*1000 #simulation time (back to fs)
                y=MSD[1][0:] #MSD(D*)
                

            #Two options give same results. ???? Do we need to do this at every step or last one would suffice??
            if 1: # compute by linear least-squares fit. This one is slightly faster.
                #A = np.vstack([x, np.ones(len(x))]).T
                #m, c = la.lstsq(A, y)[0] #m: slope is 2d*MSD
                m,c, r_value, p_value, std_err=linregress(x,y)
                #m,c, r_value, p_value, std_err=linregress(x,y-c) #y-intercept at 0 #does not work, only shifts the fit (slope is unaffected).
                r2_vals.append(r_value**2)
                p_vals.append(p_value)
                stderrs.append(std_err)

                #x = sm.add_constant(x)
                #model = sm.OLS(y,x)
                #results = model.fit()
                #m,c=results.params

                #reg = LinearRegression(fit_intercept=0, normalize=False).fit(x, y)
                #m=reg.coef_
                #c=reg.intercept_
                #print (m,c)

            else: #compute by polynomial fit
                lfit=np.polyfit(x,y,1)            
                m=lfit[0]

            tdc.append(m/2/dim*0.1)#/(args.skipSteps*dt*(st+1)))#convert A**2/fs to cm**2/s #Showul we avg over time or not? originally not. As taking the slope of MSD vs. t 1/t should be automatically included (i.e. A^2/fs).

            #TDC convergence needs revision. Now cannot detecet the changes later in the taj.
            if len(tdc)>1 and tdc[-2] > 0 and tdc[-1] >0  and not ifConv and abs(tdc[-1]-tdc[-2])/abs(tdc[-2])<=err: convSt=(st+1);ifConv=1
            #if len(tdc)>1  and tdc[-2] > 0 and tdc[-1] >0 and not ifConv and np.std(tdc)<=err: convSt=(st+1);ifConv=1  #This might become comp. expensive for large time steps.

            #end of time step loop.

        #This is prepartion for plotting the final best fit line.
        if exclZero: 
            startPt=[MSD[0][1],MSD[0][-1]] #first point is not included in the analysis.
        else:
            startPt=[MSD[0][0],MSD[0][-1]] #first point is included in the analysis.
        endPt=[m*x[1]+c,m*x[-1]+c] #End point of the best-fit line of the MSD plot.

        if convSt !=0: 
            print("TDC converged to %.1e deviation after %d steps."%(err,convSt))
            if args.discardNonConvergedTDC: #discard the part in which TDC was not converged.
                print("Discarding the part whereby TDC is not converged, as requested...")
                x=(MSD[0][convSt:])*1000 #simulation time
                y=MSD[1][convSt:] #MSD(D*)
                #A = np.vstack([x, np.ones(len(x))]).T
                #m, c = la.lstsq(A, y)[0] #m: slope is 2d*MSD
                m,c, r_value, p_value, std_err=linregress(x,y)
                tdc[-1]=m/2/dim*0.1#/(args.skipSteps*dt*(st+1))#convert A**2/fs to cm**2/s #Should we avg over time or not? originally not
                startPt=[MSD[0][convSt],MSD[0][-1]]
                endPt=[m*x[1]+c,m*x[-1]+c] 

        else: print("TDC does not seem to have converged to %.1e deviation. Please carefully inspect the TDC plot to make sure the final TDC and sigma values are reliable."%err)
        print("Final tracer diffusion coefficient, D* = %.5e cm^2/s"%tdc[-1])

        sigma=compCond(aType=args.aType,N=N,tdc=tdc[-1],T=T,Hr=Hr,ifPrint=1)
        #if sigma != 0.0: print "Conductivity at %d K, sigma=%.5e S/cm."%(T,sigma)
        #print r2_vals,sum(r2_vals),p_vals,stderrs
        print("Best-fit line (regression) statistics: R^2=%.3f, <R^2>=%.3f, <P-values>=%.2f, <StdErr>=%.3e"%(r2_vals[-1],np.mean(r2_vals),np.mean(p_vals),np.mean(stderrs)))
        if args.detailed:
            sortedMSDs=np.argsort(atomMSDs) #sorted indices
            for at in reversed(sortedMSDs):
                print("#Atom ID: %3d, MSD = %.5e A^2"%(atoms_list[at],atomMSDs[at]))


        print("Total elapsed time: %.2f sec."%( time.time()-initT))



        if 0: #args.saveGeom #now done in the getCoords function.
            initT2=time.time()
            print("Writing unwrapped coordinates to unwrapped.xyz...")
            writeCoords("unwrapped.xyz",coords_arr,aTypes,lattice)
            print("Elapsed time: %.2f sec."%( time.time()-initT2))

        #Write out the MSD data along the trajectory.
        outf=open("msd.dat","w")
        outf.write("# Final tracer diffusion coefficient, D* = %.5e cm^2/s\n"%tdc[-1])
        outf.write("# Conductivity at %d K, sigma=%.5e S/cm.\n"%(T,sigma))
        for i in range(len(MSD[0])): outf.write("%.5f %.5f\n"%(MSD[0][i],MSD[1][i]))
        outf.close()

        if args.noplot: exit()
        #####################
        # MSD plotting part #
        #####################
        #plot tdc using matplotlib
        # general options for plot
        font = {'family': 'serif', 'size': 18}
        plt.rc('font', **font)
        fig = plt.figure(figsize=(11.69, 8.27))
        #fig = plt.figure()
        gs = GridSpec(2, 1)#, width_ratios=[1, 2],height_ratios=[1, 1])
        ax1 = plt.subplot(gs[0])  #Top subplot
        ax2 = plt.subplot(gs[1])#   , sharex=ax1)


        ax2.set_xlabel('Time [ps]')
        ax2.set_ylabel('MSD [A^2]')
        ax1.set_ylabel('TDC (D*) [cm^2/s]')
        ax1.set_xticklabels([])
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))  
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,100))
        #ax2.tick_params(axis='x',which='minor',bottom='on')
        ax1.minorticks_on();ax2.minorticks_on()
        ax2.plot(MSD[0],MSD[1]) #,label="total DOS",color=(0.6, 0.6, 0.6))
        ax1.plot(MSD[0],tdc)
        #plt.legend(fancybox=True, shadow=True, prop={'size': 14})
        #fig.set_xlabel('Time step [ps]')
        #fig.set_ylabel('Mean-Square-Deviation (MSD) [A^2]')
        #plt.grid()

        #Add the best fit line, [xtart xend],[ystart yend]
        ax2.plot(startPt, endPt, 'r-') 
        #ax2.plot(lfitp, 'r-') #Add the best fit line
        #ax2.plot([0.0, endPt[0]], [0.0,endPt[1]], 'r-') #Add the best fit line

        #names = [re.sub('(\d+)', r'$_{\1}$', ref[2])           for ref in self.references] #ciuld be used for auto-subscript of numbers in stoich.
        ax1.set_title('%s (Z=%d), %d K'%(stoich,fu,T))#,fontsize=12)
        plt.subplots_adjust(hspace=0,wspace=0)
        #plt.tight_layout()

        #Add text box D* and sigma
        #textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
        textstr= 'D* = %.5e cm^2/s \n$\sigma$ = %.5e S/cm'%(tdc[-1],sigma)
        #set the props of the box
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=14,  verticalalignment='top', bbox=props)
        #axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
        #text_box = TextBox(axbox, 'Evaluate', initial=initial_text) #This is an interactive input textbox
        plt.show()

        #interactive()
        
        #print plt.isinteractive()
        #plt.ion()
        #print plt.isinteractive()

        #plt.savefig(argv[0].strip(".py") + ".pdf", format="pdf")
        #plt.savefig("diffusivity.png", format="png")
        
        
########################################
#### Velocity autocorrelation function (VACF) part ########
########################################

#This part computes the diffusivity (and then converts to conductivity) from the VACF, and also computes the IR itnensities from the vibrations along the MD sims from the FFT of VCAF.
    if args.type=='vacf' or args.type=="a": #Compute the jump-rates here.

        #OLD VERSION
        #dt defined above
        dt2=dt*args.skipSteps
        dt=dt*1e-15*args.skipSteps

        err=args.tdctol #relative change/error for VAFC(D*).
        #noSteps=(len(coords_arr))#done above
        ACF_int=np.zeros((len(coords_arr)))#noSteps
        window = choose_window(noSteps,typ=args.FFT_window_type)
        normal_vectors = np.linalg.norm(coords_arr, axis=-1) 

        
        for i in range(len(atoms_list)):
            at=atoms_list[i]
            ##print "#### shape of the normal_vectors #####", np.shape(normal_vectors)
            #print "#== The shape of the each atom array ==#", np.shape(normal_vectors[:,i])
            
            vel = calc_derivative(normal_vectors[:,at], dt) # scalar velocity of each atom aliong the traj.
            vel_vec = calc_derivative(coords_arr[:,at], dt2)

            ACF = calc_ACF(vel)
            newVACF=np.zeros((len(coords_arr)))

            #ACF = calc_ACF(np.linalg.norm(vel_vec, axis=-1))

            #ACF_int[1:]+=ACF[1:] #do not take the first step.
            #"""
            cos_t=np.zeros((len(coords_arr)))
            #for st in range(args.t0,len(coords_arr)):
            for st in range(len(coords_arr)):
                #cpos=coords_arr[st,at]
                #ipos=coords_arr[args.t0,at]
                #cos_t[st]=np.dot(cpos,ipos)/normal_vectors[st,at]/normal_vectors[args.t0,at]#(np.linalg.norm(cpos)*np.linalg.norm(ipos))
                newVACF[st]+=np.dot(vel_vec[st],vel_vec[0])
            #"""

            #ACF = calc_ACF(newVACF)

            #ACF_int[args.t0+1:]+=ACF[args.t0+1:]*ACF[args.t0]#*cos_t[args.t0+1:] #*dt2/2 #integrate over all timesteps using trapozoidal rule (diff*dt/2) (WRONG): only sum over all atoms (outer loop). #do not take the first step.
            #ACF_int[1:]+=ACF[1:] 
            ACF_int[:]+=newVACF[:] 

            yfft_at = calc_FFT(ACF, window) #original ACF
            #yfft_at = calc_FFT(newVACF, window) 

            if i == 0: #sum over all atoms included.
                yfft = yfft_at
            else:
                yfft += yfft_at

        #Get the Frouier transform of the VAFC to obtain thespectral density of states (or IR intensities). Intensitty at omega=0 should be equal to Diff. Coeff. (tdc).
        wavenumber = np.fft.fftfreq(len(yfft), dt * c )[0:int(len(yfft) / 2)] #no need for agrs.skipSteps as initially done above.
        intensity = yfft[0:int(len(yfft)/2)]
  
        #if 1: #do a convergence test for discardign the big fluctiations in the beginning.
        convSt=0; ifConv=0
        #tVACF_arr=[0.0]
        tVACF_arr=np.zeros((len(ACF_int)))
        #tVACF_arr[args.t0]=ACF_int[args.t0]
        tVACF_arr[0]=ACF_int[0]
        ms=[]
        #for j in range(args.t0+1,len(ACF_int)):#iterate over timesteps.
        for j in range(1,len(ACF_int)):#iterate over timesteps.
            tVACF_arr[j]+=tVACF_arr[j-1]+ACF_int[j]#(ACF_int[j]*dt2/len(atoms_list)/dim)
            #tVACF_arr[j]+=tVACF_arr[j-1]+(newVACF[j]*dt2/len(atoms_list)/dim)

            #x=range(args.t0,j)
            x=list(range(0,j))
            #y=tVACF_arr[args.t0:j]
            y=tVACF_arr[0:j]
            #A = np.vstack([x, np.ones(len(x))]).T
            #m, c = la.lstsq(A, y)[0] 
            m,c, r_value, p_value, std_err=linregress(x,y)

            #vacf=m*dt2
            ms.append(m)
            #print "best-fit slope of tVACF: %.5e"%m

            #if  ACF_int[j-1]!=0 and abs(ACF_int[j]-ACF_int[j-1])/abs(ACF_int[j-1]) <err: convSt=j;ifConv=1;#break
            #if  tVACF_arr[j-1]!=0 and abs(tVACF_arr[j]-tVACF_arr[j-1])/abs(tVACF_arr[j-1]) <err: convSt=j;ifConv=1;#break
            #if  j > args.t0+3 and abs(ms[-1]-ms[-2])/abs(ms[-2]) <err: convSt=j;ifConv=1;#break
            if  j > 3 and abs(ms[-1]-ms[-2])/abs(ms[-2]) <err: convSt=j;ifConv=1;#break
            #if abs (ACF_int[j]-ACF_int[j-1]) <err: convSt=j;ifConv=1;#break

        if ifConv: print("VACF converged to %.1e deviation after %d steps."%(err,convSt))
        ###
        """
        if ifConv:
            print "Discarding the non-converged part as requested."
            tVACF=sum(ACF_int[args.t0+convSt:])#/len(atoms_list) #done below
        else:
            tVACF=sum(ACF_int[args.t0:])#/len(atoms_list)
        """
        ###

        
        #x=np.arange(0,len(tVACF_arr),dt2) #simulation time
        x=list(range(len(tVACF_arr)))
        y=tVACF_arr #MSD(D*)
        #A = np.vstack([x, np.ones(len(x))]).T
        #m, c = la.lstsq(A, y)[0] 
        m,c, r_value, p_value, std_err=linregress(x,y)

        vacf=m*dt2 #A^2/fs^3*fs
        #print "best-fit slope of tVACF: %.5e"%m
 
        #vacf=tVACF  #these two give the same, when no steps discarded. #this can give neg. diffusivity values !!
        #vacf=ACF_int[-1]*float(1.0/dim/len(atoms_list))*dt2/2
        #newVACF*=float(1.0/dim/len(atoms_list))
        #vacf=tVACF_arr[-1]/(dt2*noSteps) #this can give neg. diffusivity values !!
        #tdc=np.trapz(newVACF[args.t0:],dx=dt2)*1e-15#*dt2*args.skipSteps #? why covnersion factor?? #args.skipSteps needed for removing the ffect of step skiping.
        #tdc1=np.trapz(ACF_int[args.t0:],dx=1)*dt2/dim/len(atoms_list)/noSteps*args.skipSteps*1e-7 #? why conversion factor?? #args.skipSteps needed for removing the ffect of step skiping. This does not work.        #ALSO AVG over no of TIMESTEPS.
        tdc1=np.trapz(ACF_int[0:],dx=1)*dt2/dim/len(atoms_list)/noSteps*args.skipSteps*1e-7 #? why conversion factor?? #args.skipSteps needed for removing the ffect of step skiping. This does not work.        #ALSO AVG over no of TIMESTEPS.
        print("Final self-diffusion coefficient using integral of total VACF, D* = %.5e cm^2/s"%(tdc1))     
        sigma1=compCond(aType=args.aType,N=N,tdc=tdc1,T=T,Hr=Hr,ifPrint=1)
        #if sigma1 != 0.0: print "Conductivity at %d K, sigma=%.5e S/cm."%(T,sigma1)
       
        tdc2=vacf*0.1*dt2/dim/len(atoms_list)/noSteps/args.skipSteps#convert A**2/fs to cm**2/s #CHECK THIS!!!
        #tdc=vacf*0.1#*args.skipSteps/#convert A**2/fs to cm**2/s #CHECK THIS!!!+
        print("Final self-diffusion coefficient using best-fit of total VACF, D* = %.5e cm^2/s"%(tdc2)) 
        sigma2=compCond(aType=args.aType,N=N,tdc=tdc2,T=T,Hr=Hr,ifPrint=1)
        #if sigma2 != 0.0: print "Conductivity at %d K, sigma=%.5e S/cm."%(T,sigma2)
     

        tdc3=intensity[0] /dim#*0.1*dt2#*1e3#*dt*1e15/dim/len(atoms_list)  #D_G=G(0), where G(omega) is the FFT of VACF, spectral density of states, as decribed in Andriyesvski et al. Mat. Chem. Phys. 2017 #convert A**2/fs to cm**2/s   #This somewhat works for every case!! This gives better results mathcing MSD one.
        print("Final self-diffusion coefficient using spectral DOS (FFT of VACF), G(omega=0) = D* = %.5e cm^2/s (sensitive on the choice of timestep-skipping.)"%(tdc3))       
        sigma3=compCond(aType=args.aType,N=N,tdc=tdc3,T=T,Hr=Hr,ifPrint=1)

        print("Total elapsed time: %.2f sec."%( time.time()-initT))


        #Add visualization and save results here!!
 
        init=args.t0*dt*1e12#*args.skipSteps burda olmali ???
        t=np.linspace(init,init+dt*1e12*noSteps,noSteps)
 
        outf='vacf.dat'
        title = ("# Timestep", "VACF_integrated", "# ps", "a.u.")
        #arr=tVACF_arr
        #arr=newVACF
        arr=ACF_int/len(atoms_list)
        with open(outf, "w") as fw:
            np.savetxt(outf, np.c_[t,arr],
                       fmt="%10.5f %15.5e",
                       header="{0:>10}{1:>16}\n{2:^11}{3:^20}".format(*title),
                       comments='')

        outf='IR_intensities.dat'
        title = ("# Wavenumber", "IR Intensity", "# cm^-1", "a.u.")
        with open(outf, "w") as fw:
            np.savetxt(outf, np.c_[wavenumber,intensity],
                       fmt="%10.5f %15.5e",
                       header="{0:>10}{1:>16}\n{2:^11}{3:^20}".format(*title),
                       comments='')


        ###################
        #### VACF plotting part ####
        ###################
        plt.figure(figsize=(11.69, 8.27))
        #set the props of the text boxes
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)

        #tmp=newVACF
        tmp=ACF_int/len(atoms_list)
        #tmp=tVACF_arr
        L2 = np.arange(len(tmp))
        #L2=t
        ax1=plt.subplot(3,1,1)
        ax1.plot(L2, tmp, color='red', linewidth=1.5)
        ax1.axis([0, len(tmp), np.min(tmp), np.max(tmp)], fontsize=15)
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,1))
        plt.xlabel("Data Points", fontsize=12)
        #plt.ylabel("VACF (a.u.)", fontsize=12)
        plt.ylabel("Avg. VACF", fontsize=12)
        #Add text box D* and sigma
        textstr= 'D* = %.5e cm^2/s \n$\sigma$ = %.5e S/cm'%(tdc1,sigma1)
        ax1.text(0.75, 0.25, textstr, transform=ax1.transAxes, fontsize=10,  verticalalignment='top', bbox=props)

        #tmp=tVACF_arr
        #tmp=ACF_int
        tmp=ms
        L2 = np.arange(len(tmp))
        #L2=t
        #plt.subplot(3,1,2)
        ax2=plt.subplot(3,1,2)
        ax2.plot(L2, tmp, color='red', linewidth=1.5)
        ax2.axis([0, len(tmp), 1.1*np.min(tmp), 1.1*np.max(tmp)], fontsize=15)
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,1))
        #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))

        plt.xlabel("Data Points", fontsize=12)
        plt.ylabel("Slope(t-VACF)", fontsize=12)
        
        #Add text box D* and sigma
        textstr= 'D* = %.5e cm^2/s \n$\sigma$ = %.5e S/cm'%(tdc2,sigma2)
        ax2.text(0.75, 0.25, textstr, transform=ax2.transAxes, fontsize=10,  verticalalignment='top', bbox=props)

        ax3=plt.subplot(3,1,3)
        ax3.plot(wavenumber, intensity, color='black', linewidth=1.5)
        ax3.axis([0 , 2000, #len(wavenumber),#np.max(wavenumber)*1.1,
                  np.min(intensity), 1.1*np.max(intensity[:4001])],fontsize=15)
        ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,1))
        plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=12)
        plt.ylabel("Spectral DOS", fontsize=12)

        ax1.set_title('%s (Z=%s), %d K'%(stoich,fu,T))#,fontsize=12)

        #Add text box D* and sigma
        textstr= 'D* = %.5e cm^2/s \n$\sigma$ = %.5e S/cm'%(tdc3,sigma3)
        ax3.text(0.75, 0.90, textstr, transform=ax3.transAxes, fontsize=10,  verticalalignment='top', bbox=props)



        plt.subplots_adjust(hspace = 0.5)
        plt.show()

########################################
#### Mean Jump Rates (MJR) part ########
########################################

    if args.type=="jump" or args.type=="a": #Compute the jump-rates here.
        from spglib import *
        #First deteremine the crystalline sites (either from initial Li/Na atomic positions from .xyz  or from Wykoff positons retrieved from external file).
        tol=args.tol

        """
        aTypes=[]
        initCoords=[]
        #for crd in coords[args.t0]:  #Set the initial frame as the reference.
        for crd in coords[0]:  #Set the initial frame as the reference.
            aTypes.append(crd[0])
            initCoords.append(crd[1:])
        atoms=Atoms(aTypes,positions=initCoords,cell=lattice,pbc=[1,1,1]) 
        """
        atoms=Atoms(aTypes,positions=coords_arr[0],cell=lattice,pbc=[1,1,1]) #Set the initial frame as the reference. #aTypes was created in the beginning.


        if not args.initCrysFile:
            atoms_ref=atoms
            toln=tol
        else: #if an input file for crsy. site info is provided.
            if args.initCrysFile.split("/")[-1]=="POSCAR": fm="vasp"
            else: fm=None
            atoms_ref=ase.io.read(args.initCrysFile,format=fm)
            toln=1e0

        if args.ifPrim: #Find the primitive cell of the ref. str.
            scaled_positions= atoms_ref.get_scaled_positions()
            cell=(atoms_ref.cell, atoms_ref.get_scaled_positions(), atoms_ref.numbers)
            print(("Using the primitive cell of the reference structure with SG=%s"%(get_spacegroup(cell,symprec=args.tol))))
            lattice, scaled_positions, numbers = find_primitive(cell, symprec=tol)
            atoms_ref=Atoms(symbols=numbers,cell=lattice,scaled_positions=scaled_positions,pbc=True)

        cell=(atoms_ref.cell, atoms_ref.get_scaled_positions(), atoms_ref.numbers) ##!!! This should be scaled positions.

        data = get_symmetry_dataset(cell, symprec=tol, angle_tolerance=-1.0, hall_number=0)#or toln?
        #print data
        sg=get_spacegroup(cell, symprec=tol) #or toln?
        print("Space group= %s with tol=%.2e"%(sg,tol))


        #Compare the input crys. str and compare it with the reference in an attempt to determine the number of repettitions. If ound, then repeat the ref. cell to get the ref. sites in the extended cell. 
        rpt=compCells(atoms.cell,atoms_ref.cell)
        rpt_no=rpt[0]*rpt[1]*rpt[2]
        #if args.verb:atoms_ref.write("init_ref.vasp",format="vasp")
        if args.verb:atoms_ref.write("init_ref.xyz",format="xyz")
        atoms_ref=atoms_ref.repeat(rpt)
        #if args.verb:atoms_ref.write("init_ref_extended.vasp",format="vasp")
        if args.verb:atoms_ref.write("init_ref_extended.xyz",format="xyz")

        if args.verb: print("Repeating the ref. crys. str. by %dx%dx%d to match the input str."%(rpt[0],rpt[1],rpt[2]))

        #Check if the number of atoms match with extended ref. str.
        if len(atoms) != len(atoms_ref):
            print("WARNING: No of atoms in the input and reference structures do not match !! ")
        #Determine the equivalent atoms and sites.
        equiSites={}    
        if data is not None:
            for i in range(len(data['equivalent_atoms'])):
                ea=str(data['equivalent_atoms'][i])
                if ea not in equiSites: equiSites[ea]=[data['wyckoffs'][i],i]
                else: equiSites[ea].append(i)

            #Correct the keys.
            keys=list(equiSites.keys())
            keys.sort()
            eS=dc(equiSites)
            equiSites={}
            for i in range(len(keys)):
                key = keys[i]
                equiSites['equi_site%d'%(i+1)]=eS[key]
        else: 
            print("No equivalent positions. All selected atoms will be treated separately.")
            equiSites={}
            for i in range(len(atoms_list)):
                equiSites['equi_site%d'%(i+1)]=['a',i]
            #equiSites['equi_site1'].extend()) #Atom id of al latoms.
        if args.verb:print(equiSites)

        eSites={}
        keys=list(equiSites.keys())
        keys.sort()
        cSites=[]
        pos=atoms_ref.get_positions() #Use scaled pos instead ?? Nope
        init_wyckoffs={}  #Not used anywhere!!
        #crd=atoms_ref.get
        for key in keys:
            val=equiSites[key]
            ##cst=[x for x in atoms_ref.get_positions()]
            if '%d%s'%(len(val)-1,val[0]) not in eSites: eSites['%d%s'%(len(val)-1,val[0])]=val[1:]
            else: eSites['%d%s'%(len(val)-1,val[0])].extend(val[1:])

        print("The following equivalent sites were extracted from the reference crystal file: ")    
        
        for eS in sorted_nicely(list(eSites.keys())):
            print("%s: %s"%(eS,sorted(eSites[eS]))) #This is not used anywhere else in the code. Just here for informing.
        print()

        #Use the extended cell from the reference structure to assign the Wyckoffs of the whole set of reference crys. sites.
        for key in keys:
            val=equiSites[key]
            k='%d%s'%(len(val)-1,val[0])
            for vl in val[1:]:
                for ii in range(rpt_no):
                    ind=int(vl+ii*len(atoms_ref)/rpt_no)

                    if 0: #Check for two close crystallien sites to prevent doule counting.
                        for cs in cSites:
                            other_pos=cs[0]
                            #print "%d %.3e"%(ind,minDist(pos[ind],other_pos,atoms_ref.cell)) ,
                            if minDist_ASE(pos[ind],other_pos,atoms_ref.cell) <5e-1: print("Site %d too close to previous site %s"%(ind,cs[0])); continue

                    cSites.append([pos[ind],k,atoms_ref[ind].symbol,ind])#.tolist())

        #Filter out the unwanted crystalline sites (i.e. discluded from jump tracking) here!
        if args.allowedSites or args.limitJumps:
            cSites2=[]
            for cS in cSites:
                if args.allowedSites and cS[1] in args.allowedSites:
                    cSites2.append(cS)
                elif args.limitJumps and cS[2] in args.aType:
                    cSites2.append(cS)
            del cSites
            cSites=dc(cSites2)
            del cSites2

        if len(cSites)==0: print("Selected crys. site types (%s) are not found in the ref. structure. Terminating..."%(args.allowedSites));exit()

        if args.verb: 
            print("Coordinates and Wyckoff letters of the reference crystalline sites: ")
            for cS in cSites:         print(cS)

    #Determine the closest nN crystalline sites (neighbours) to each atom.
        closestSites=[]
        #trackSites=[] #to be used below, tracks which atom is in which site along the MD trajectory. #To speed up, consider not storing this.
        #trackJumps=[]

        NNB=0
        if not args.Nneighbours=="all": 
            if int(args.Nneighbours) > len(cSites): NNB=12
            else: 
                try: NNB=int(args.Nneighbours)
                except:NNB=12
        else: 
            NNB=len(cSites)

        print("Number of closest neighbours to consider in jump rate calculation: %d"%NNB)


        #a=0.0 #avg. distance btw. init. crys. sites.
        #for site in cSites:#Only consider the list of selected atoms.
        for at in atoms_list:
            #site=cSites[at]
            #site=coords[args.t0][at] #this is needed when initial crys sites are read from external file. This would be equivalent to cSites[at] when input.xyz is used for initial crys. sites.
            site=coords_arr[0][at] #this is needed when initial crys sites are read from external file. This would be equivalent to cSites[at] when input.xyz is used for initial crys. sites.
            dists={}
            wycks={}
            #trackSites.append([]) 
            #trackJumps.append(0)
            for site2 in cSites:
                wyck=site2[1]
                site2=site2[0].tolist()
                x=minDist(site,site2,lattice) #do it with liinalg.norm or ASE atoms dist tool. works faster than _ASE, weird ??? CHECK again
                #x=minDist_ASE(site,site2,lattice) 
                #a += x
                if str(x) in dists: #key:distance
                    dists[str(x)].append(site2)#; print x, dists[str(x)]
                    wycks[str(x)].append(wyck)  #Hata burdaydi !! extend idi
                else: 
                    dists[str(x)]=[site2]
                    wycks[str(x)]=[wyck] #bu [] siz mi olmali?
            
            #print "wycks: ",wycks

            keys=[float(x) for x in list(dists.keys())]
            keys.sort()
            tmp=[[],[],[]]
            if args.Nneighbours=="all" or len(keys)<NNB: NNB=len(keys)#;print "Number of neighbours: %d"%NNB
            for i in range(0,NNB):
                key=str(keys[i])  #key: distance; dists[str(key)]: crys sites. #CHECK if the indexing is correct !!!i is for NNB not for keys.
                #for j  in range(len(dists[str(key)])): #account for multiple sites with same distance (unlikely but can occur). Not needed, it isalready done wout this.
                sit=dists[key]
                tmp[0].append(sit) #crys. site coords.
                tmp[1].append(float(key)) #distance
                tmp[2].extend(wycks[key]) #wyckoffs of these sites.       
            closestSites.append(tmp)
            if args.verb: print("Atom %4d neighbours (%d): %s"%(at,len(closestSites[-1][-1])," ".join(closestSites[-1][-1])))  #initial assignments of closest sites are incorrect.??

        #a /= (len(atoms_list)*len(cSites)) #avg. distance btw. init. crys. sites.
        #exit()

        init_dists=[]
        if args.verb: 
            print("\nInitial assignment of the atoms to the reference crystalline sites.")
            print("Id   Type  Crys. Site  Distance [A].")
        for i,at in enumerate(atoms_list):
            if args.verb: print("%-4d %-5s %-12s %-.5f"%(at+1,aTypes[at],closestSites[i][2][0],float(closestSites[i][1][0])))
            init_dists.append(closestSites[i][1][0])
        
        print("Avg. initial distance of atoms to the respective ref. crystalline sites: %.5f +- %.5f"%(mean(init_dists),stddev(init_dists))) #FIX this for the filtered case.

        #print closestSites
        trackSites={}#to hold the number of jumps between diverse crys. sites.
        trackJumps=[ 0 for i in range(len(atoms_list)) ]
        nparts=args.nprocs #16 #partition big trajectories.

        if args.nprocs != 1 or args.nprocs== None: 
            if args.parAtoms: #parallelize over the seleted atoms set.
                print("No of chunks: ",nparts)
                print("Chunk size: %d atoms"%(len(atoms)/nparts))
                print("No of processes: ",args.nprocs)

                #trackJumps,trackSites=getJumps_atoms(atoms_list,args.t0,noSteps,closestSites,lattice,coords,nprocs=args.nprocs,verb=args.verb) #parallesiation done in the function.
                trackJumps,trackSites=getJumps_atoms(atoms_list,0,noSteps,closestSites,lattice,coords_arr,nprocs=args.nprocs,verb=args.verb) #parallesiation done in the function.

            else: #parallelize over the coord set.
                #chunk=int((len(coords)-args.t0)/nparts)
                chunk=int(len(coords_arr)/nparts)
                print("No of chunks: ",nparts)
                print("Chunk size: %d time steps"%chunk)
                print("No of processes: ",args.nprocs)

                #trackJumps,trackSites=getJumps_coords(atoms_list,args.t0,chunk,closestSites,lattice,coords,nprocs=args.nprocs,verb=args.verb) #parallesiation done in the function.
                trackJumps,trackSites=getJumps_coords(atoms_list,0,chunk,closestSites,lattice,coords_arr,nprocs=args.nprocs,verb=args.verb) #parallesiation done in the function.
                    
        else: #serial run.
            print("Running a serial MJR analysis.")
            #trackJumps,trackSites=getJumps_serial(atoms_list,args.t0,noSteps,closestSites,lattice,coords)
            trackJumps,trackSites=getJumps_serial(atoms_list,0,noSteps,closestSites,lattice,coords_arr)

        if args.verb:        print(trackJumps);        print(trackSites)

        trackSites=combineJumps(trackSites) #Combine same jump types in reverse order 
        keys=list(trackSites.keys())
        keys.sort()

        avgs,a=getAvgCrysSiteDist(cSites,lattice)
        totDj=0.0
        taos={}
        for k in keys:
            #site=trackSites[k]
            taos[k]=float(trackSites[k])/len(atoms_list)/float(dt*args.skipSteps*noSteps)*1e15 #fs^-1 to s^-1.
            Dj=taos[k]*(avgs[k]**2)/2/dim*angtocm**2  #Check this!!
            totDj+=Dj
            print("#jumps(%s)= %5d; tao(%s)= %.2e s^-1, Dj= %.2e cm^2/s, a=%.2f A, sigma= %.2e S/cm"%(k,trackSites[k],k,taos[k], Dj,avgs[k],compCond(aType=args.aType,N=N,tdc=Dj,T=T,Hr=Hr,ifPrint=0)))

        #tao=float(sum(trackJumps))/len(atoms_list)/(dt*args.skipSteps*len(coords))*1e15 #fs^-1 to s^-1.
        tao=sum(taos.values())
        print("\nOverall mean jump rate (tao)= %.2e s^-1"%tao)

        if 0:
            print("Overall average distance btw crys. sites (a)= %.4f A"%a)
            totDj=tao*(a**2)/2/dim*angtocm**2 #diffusivity [cm^2/s] using Einstein-Smoluchowski eqn. ang^2/s to cm^2/s

        print("Overall %s Diffusivity (Dj)= %.5e cm^2/s."%(", ".join(args.aType),totDj)) #TODO compute as sum of individual contobutions.

        #sigma=N*e**2*Z[args.aType]**2*Dj/(V*kb*T*angtocm**3)  #Using Nernst-Einstein equation  #CHECK the unit conversion!!!
        sigma=compCond(aType=args.aType,N=N,tdc=totDj,T=T,Hr=Hr,ifPrint=1)
        #print "Conductivity at %d K, sigma=%.5e S/cm."%(T,sigma)

        if args.detailed:
            sortedJumps=np.argsort(trackJumps) #sorted indices
            print('Individual jump rates, i.e. mobility of each ion:')
            totDj=0.0
            for at in reversed(sortedJumps):
                 tao=float(trackJumps[at])/float(dt*args.skipSteps*noSteps)*1e15 #fs^-1 to s^-1. #no need to avg over atoms.
                 Dj=tao*(avgs[k]**2)/2/dim*angtocm**2
                 totDj+=Dj
                 print("#Atom ID: %3d, #jumps= %5d; tao= %.2e s^-1, Dj= %.2e cm^2/s"%(atoms_list[at],trackJumps[at],tao,Dj))

        print("Elapsed time: %.2f sec."%( time.time()-initT))

    ###########################
    # Radial Pair Disctribution (RDF) related part  #
    ###########################

    if args.type=="rdf" or args.type=="a":
        #args.skipSteps have generally only a smoothening/coarsening effect on the RDF. (CHECKED!)
        #Get these from the user !!
        rmin=0.
        rmax=7.
        bin_width=0.10#delta_r
        nbins=None
        norm_by_V=1 # Normalize by volume of the spherical shell. (should be on)
        dists={};        init_dists={}
        for i,at1 in enumerate(atoms_list): #can be parallelised over atoms.
            print(i, );stdout.flush()
            #for j in range(i,len(atoms_list)):
            for at2 in range(len(aTypes)):
                if at2==at1: continue
                key1="%s-%s"%(aTypes[at1],aTypes[at2])                
                key2="%s-%s"%(aTypes[at2],aTypes[at1])
                for st in range(noSteps):
                    #x=minDist(coords_arr[st][at1],coords_arr[st][at2],lattice) #this matches with VMD using PBC (nly the peaks at longer dist 7-9 A are lower a. Very slow!
                    x=minDist_ASE(coords_arr[st][at1],coords_arr[st][at2],lattice) #Using the find_mic distance implemented in ASE.  (Use this when using wrapped coords).
                    #x=dist(coords_arr[st][at1],coords_arr[st][at2]) #this matches with VMD not using PBC. #This should be used jointly  with unwrapped coordinates, otherwise no meaning for PBC systems. Much faster than other two methods.
                    #if x>rmax or x<rmin:continue #No need to store the out-of-range  values. VMD matches when this is commented out !!!
                    if key1 in dists: dists[key1].append(x)
                    elif key2 in dists: dists[key2].append(x)
                    else: dists[key1]=[x]
                    if st==0:
                        if key1 in init_dists: init_dists[key1].append(x)
                        elif key2 in init_dists: init_dists[key2].append(x)
                        else: init_dists[key1]=[x]
        print()

        #fig=plt.figure()
        fig = plt.figure(figsize=(8.0, 6.0))
        #ax = fig.add_subplot(111)
        #fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

        gs = GridSpec(2, 1)#, width_ratios=[1, 2],height_ratios=[1, 1])
        ax1 = plt.subplot(gs[0])  #Top subplot
        ax2 = plt.subplot(gs[1])#   , sharex=ax1)


        keys= sorted(dists.keys())
        total=[];total_ref=[]
        cc=-1
        #cl=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        cl=[ 'g', 'C1','C4','C5','C6','C7','C8', 'c', 'm', 'y', 'w']
        for key in keys:
            cc+=1
            total.extend(dists[key])
            r,g_r,crd= compute_rdf(dists[key],V,r_range=(rmin,rmax), bin_width=bin_width, n_bins=nbins,norm_by_V=norm_by_V)
            if args.verb:print(key,r,g_r,crd)
            ax2.plot(r,g_r,color=cl[cc%len(cl)])
            ax1.plot(r,crd,color=cl[(cc)%len(cl)],label=key)


            outf="rdf-%s.dat"%(key)
            title = ("# radius", "g(r)", "# A", "a.u.")
            with open(outf, "w") as fw:
                np.savetxt(outf, np.c_[r,g_r],
                           fmt="%10.5f %15.5e",
                           header="{0:>10}{1:>16}\n{2:^11}{3:^20}".format(*title),
                           comments='')
            
            if 1: #use initial frame as the ref.
                total_ref.extend(init_dists[key])
                r,g_r,crd= compute_rdf(init_dists[key],V,r_range=(rmin,rmax), bin_width=bin_width, n_bins=nbins,norm_by_V=norm_by_V)
                if args.verb:print(key+"_ref",r,g_r,crd)
                if 0: label=key+"_ref"
                else:label='_nolegend_'
                ax2.plot(r,g_r,color=cl[cc%len(cl)],linestyle='dashed') #,label
                ax1.plot(r,crd,color=cl[(cc)%len(cl)],linestyle='dashed',label=label)

                outf="rdf-%s-ref.dat"%(key)
                title = ("# radius", "g(r)", "# A", "a.u.")
                with open(outf, "w") as fw:
                    np.savetxt(outf, np.c_[r,g_r],
                               fmt="%10.5f %15.5e",
                               header="{0:>10}{1:>16}\n{2:^11}{3:^20}".format(*title),
                               comments='')
        cc+=1

        r,g_r,crd= compute_rdf(total,V,r_range=(rmin,rmax), bin_width=bin_width, n_bins=nbins,norm_by_V=norm_by_V)
        if args.verb:print("total",r,g_r,crd)
        ax2.plot(r,g_r,color='k',label='_nolegend_')
        ax1.plot(r,crd,color='k',label='Any')

        """
        r,g_r,crd= compute_rdf(total_ref,V,r_range=(rmin,rmax), bin_width=bin_width, n_bins=nbins,norm_by_V=norm_by_V)
        ax1.plot(r,crd,color=cl[(cc)%len(cl)],linestyle='dashed')
        """

        outf="rdf-total.dat"
        title = ("# radius", "g(r)", "# A", "a.u.")
        with open(outf, "w") as fw:
            np.savetxt(outf, np.c_[r,g_r],
                       fmt="%10.5f %15.5e",
                       header="{0:>10}{1:>16}\n{2:^11}{3:^20}".format(*title),
                       comments='')

        print("Elapsed time: %.2f sec."%( time.time()-initT))
        
        ##############
        #Plotting of RDF data #
        ##############
        #plt.legend()
        ax1.legend(fancybox=True, shadow=True, prop={'size': 11},loc='upper left')
        ax1.set_title('%s (Z=%s), %d K'%(stoich,fu, T))#,fontsize=12)
        plt.xlabel(" Radius (A)", fontsize=12)
        ax1.set_ylabel("Coordination(r)", fontsize=12)
        ax2.set_ylabel("g(r)", fontsize=12)
        ax2.minorticks_on();#ax2.minorticks_on()
        ax2.tick_params(axis='x',which='major',bottom='on', top='on') #works
        ax2.tick_params(axis='x',which='minor',bottom='on', top='on')

        plt.subplots_adjust(hspace=0,wspace=0)
        plt.show()





    ######################################################################
    # Van Hove Correlation of ionic motion using PyMatGen implementation #
    ######################################################################
    #see http://guide.materialsvirtuallab.org/pymatgen-diffusion/pymatgen_diffusion.aimd.html  for more information.

    if args.type=="vanHove" or args.type=="a":
        import pymatgen #as pmg
        #from pymatgen.io.ase import *
        import pymatgen.io.ase
        from pymatgen.core.structure import Structure
        import pymatgen_diffusion as pd #this needs to be installed separately (i.e. pip3 install pymatgen_diffusion)
        #from pymatgen_diffusion import aimd
        from pymatgen_diffusion.aimd import van_hove,pathway #Takes time to load !!
        from pymatgen.analysis.diffusion_analyzer import DiffusionAnalyzer as DA

        #Using the PYMATGEN built-in tools
        init_str=pymatgen.io.ase.AseAtomsAdaptor.get_structure(coords_arr[0])
        structures=[pymatgen.io.ase.AseAtomsAdaptor.get_structure(i) for i in coords_arr] #rename it to atoms_arr
        
        print("Elapsed time: %.2f sec."%( time.time()-initT))

        font = {'family': 'serif', 'size': 14}
        plt.rc('font', **font)
        fig = plt.figure(figsize=(11.69, 8.27))       
 
        
        if 0: 
            ea=van_hove.EvolutionAnalyzer(structures, rmax=8., step=args.skipSteps, time_step=args.timeStep)
            ea.plot_atomic_evolution(args.aType[0], direction='a', df=None) #works !!
            ea.plot_atomic_evolution(args.aType[0], direction='b', df=None) #works !!
            ea.plot_atomic_evolution(args.aType[0], direction='c', df=None) #works !!

        #RDF=van_hove.RadialDistributionFunction(structures, indices, reference_indices, ngrid=101, rmax=10.0, cell_range=1, sigma=0.1)
        
        #DiffusionAnalyzer(structure, displacements, specie, temperature, time_step, step_skip, smoothed='max', min_obs=30, avg_nsteps=1000, lattices=None) #For traditional analysis, use smoothed=False and weighted=False.
        da=DA.from_structures(structures, specie=args.aType[0], temperature=T, time_step=args.timeStep, step_skip=args.skipSteps, initial_disp=None, initial_structure=None)
        if 0: 
            fig=da.get_framework_rms_plot(plt=plt, granularity=200, matching_s=None)
            fig.show()

        if 1: #plot MSD data.
            da.plot_msd(mode='default') #mode (str) – Can be “default” (the default, shows only the MSD for the diffusing specie, and its components), “ions” (individual square displacements of all ions), “species” (mean square displacement by specie), or “mscd” (overall mean square charge displacement for diffusing specie).
            da.plot_msd(mode='ions')
            da.plot_msd(mode='mscd') #charge displacement !!

        s=da.get_summary_dict(include_msd_t=1, include_mscd_t=1) #gives
        #['D', 'D_sigma', 'D_charge', 'D_charge_sigma', 'S', 'S_sigma', 'S_charge', 'D_components', 'S_components', 'D_components_sigma', 'S_components_sigma', 'specie', 'step_skip', 'time_step', 'temperature', 'max_framework_displacement', 'Haven_ratio', 'msd', 'msd_components', 'dt', 'mscd']
        print (' D= %.5e\n D_sigma=%.5e\n D_charge=%.5e\n S=%.5e\n S_sigma=%.5e\n S_charge=%.5e\n Max Framework Disp= %.5f\n Haven Ratio (H_R)=%.5f\n'%(s['D'], s['D_sigma'],s['D_charge'],s['S'],s['S_sigma'],s['S_charge'],s['max_framework_displacement'], s['Haven_ratio']))

        #pda=pathway.ProbabilityDensityAnalysis(structure, trajectories, interval=0.5, species=('Li', 'Na'))
        pda=pathway.ProbabilityDensityAnalysis.from_diffusion_analyzer(da, interval=0.2, species=args.aType) #interval (float) – the interval between two nearest grid points (in Angstrom), def: 0.5 A.
        pda.to_chgcar(filename='CHGCAR.vasp')
        if 0: 
            try:
                pda.generate_stable_sites(p_ratio=0.25, d_cutoff=1.0)
                print(pda.stable_sites)
            except: None

        soa=pathway.SiteOccupancyAnalyzer.from_diffusion_analyzer(coords_arr[0].get_scaled_positions(), da, species=args.aType)
        #print (soa.get_average_site_occupancy(range(noAtoms)))
        print ("Average site occupancy: %.2f"%soa.get_average_site_occupancy([i.index for i in coords_arr[0] if i.symbol in args.aType]))
        
        vha=van_hove.VanHoveAnalysis(da, avg_nsteps=50, ngrid=101, rmax=8.0, step_skip=args.skipSteps, sigma=0.1, cell_range=1, species=args.aType, reference_species=None, indices=None)  #skip 20 steps; takes longer than others.

        fig=vha.get_3d_plot(figsize=(12, 8), type='distinct') #this is the published one. #type: 'distinct' or 'self'
        #fig.show()

        #vha.get_1d_plot(mode='distinct', times=range(0.,noSteps*args.timeStep*args.skipSteps, args.timeStep*args.skipSteps), colors=None)
        total_time=float(noSteps*args.timeStep*args.skipSteps/1000.) #in ps
        #times=[round(i,1) for i in np.arange(0., total_time, total_time/4.)]

        #from decimal import *;getcontext().prec = 1
        #times=[float(Decimal('%.1f'%i)) for i in np.arange(0.0, total_time, total_time/4.0)]
        #times=[round(i,1) for i in np.arange(0.0, total_time, total_time/4.0)]
        #times.append(total_time)
        #times=np.arange(0., total_time, total_time/4.)

        #times=[ float(int(total_time/5.0*i)) for i in range(6)]
        times=[0.0,1.0,2.0,3.,4.,5.0,7.5,10.0,total_time]
        print(times)
        fig=vha.get_1d_plot(mode='distinct', times=times, colors=None)
        fig.legend()
        fig.show()

    if args.type=="tort" or args.type=="a":
        import crystal_torture
        from crystal_torture.pymatgen_interface import *
        import pymatgen #as pmg
        import pymatgen.io.ase

        #Using the PYMATGEN built-in tools
        init_str=pymatgen.io.ase.AseAtomsAdaptor.get_structure(coords_arr[0])
        structures=[pymatgen.io.ase.AseAtomsAdaptor.get_structure(i) for i in coords_arr] #rename it to atoms_arr
        
        print("Elapsed time: %.2f sec."%( time.time()-initT))

        #print(init_str)

        graph = graph_from_structure(structure=init_str,rcut=4.0,elements={'Li'})#args.aType)#(structure, rcut, elements)

        print(len(graph.clusters))
        for cluster in graph.clusters:
            print("Cluster: (len) (indices in cluster))")
            print( cluster.return_uc_indices())

        print([cluster.periodic for cluster in graph.clusters])
        print('{} {}\n'.format("Frac Perc:",graph.return_frac_percolating()))#Calculates the fraction of nodes in the graph that are in a periodic cluster
        #graph.torture_py()
        graph.torture()

        for no,cluster in enumerate(graph.minimal_clusters):
            print('{}{}{}\n'.format("****** Cluster: ",no, " ********"))
            print('{} {}\n'.format("Size:",cluster.size))
            print('{} {}\n'.format("Tortuosity:",cluster.tortuosity))
            print('{} {}\n'.format("Periodicity:",cluster.periodic))

        print(graph.tortuosity)
        graph.output_clusters(fmt='poscar',periodic=False) #periodic=1 only periodic (percolating) clusters.
        graph.return_periodic_structure(fmt='poscar')

        for i in range(0,noSteps,args.skipSteps):
            graph = graph_from_structure(structure=structures[i],rcut=4.0,elements={'Li'})
            graph.torture()
            ctime=float(i*args.timeStep*args.skipSteps/1000.) #in ps
            mc=graph.minimal_clusters
            try:print("%5.1f %2d %2d %5.2f "%(ctime,len(mc),mc[0].periodic,mc[0].tortuosity))
            except:None

    exit()
    
    #End of the main code
    
################
# Deleted Parts #
################


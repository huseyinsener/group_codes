#!/usr/bin/env python3


import os,time,re,math,argparse
from os import system,popen,chdir,getcwd,getenv,putenv,listdir,environ

from sys import exit,stdout,argv
from copy import deepcopy

import numpy as np
from ase import Atoms,Atom
import ase.io
from ase.visualize import view
from ase.build import sort
from spglib import *
import subprocess as sp
from subprocess import Popen,PIPE,DEVNULL,run,CompletedProcess,call # as popen # check_output

from sys import argv,exit,version_info

import os.path
from os import popen,environ,system#,popen4
#from sklearn.utils.extmath import cartesian
from itertools import combinations#,product,permutations
import operator as op
from functools import reduce

from ase.calculators.lj import LennardJones as LJ
from ase.calculators.morse import MorsePotential as MP
import ase.calculators.castep
from ase.calculators.vasp import Vasp #,Vasp2 #requires ase >= 3.17.0

import random,string
#import filecmp #Compare files
import ase.spacegroup

from spglib import find_primitive,standardize_cell,get_spacegroup #, niggli_reduce #niggli_reduce from ASE-tools conflicts with that from spglib.
#from ase.spacegroup import Spacegroup,get_spacegroup

from ase.build import niggli_reduce

from tqdm import tqdm # to show progress bars for the for loops, but makes them slower.
from p_tqdm import * #p_imap, p_map,p_umap, p_uimap #u:unordered, i:iterator #pip install p_tqdm --user     #https://pypi.org/project/p-tqdm/
#from collections import defaultdict

#import glob


from subprocess import Popen,PIPE # as popen # check_output
from sys import stdout,version_info

"""
#To-do list:
-Option to run multiple configs at each step in parallel (rather than sequential).
-Support for multiple dopant types
-Check for the size and atom number of the reference cell with input cell.
-Check for the symm-equivalent configs before or after the structure modification (i.e. Li deletion)??
-ADD PREOPTIMISE INPUT GEOM OPTION.

Known issues:
-the interstital addition not working properly when using a ref structure smaller than the actual input structure
-when using '-spick S' option (picking the highest symmetry structure at each sequential step), configs are ordered in decreasing order of energy; must be vice versa, as now the highest energy structures with highest smmetry makes it to the next step.

-(DONE) Support restarting !!! 
-(DONE) Hubbard U params through ASE
-(DONE) magmoms through ASE
-(DONE) Automate the sequential doping, choose for min-energy and high-symm structures at each step (Make code modular)  DONE
-(DONE) Add the config entropy to the computed CASTEP energies ! DONE
-(DONE) Option for CASTEP energies.  DONE
"""

def handle_magmoms(atoms, magmoms):
    """
    The magamoms variable should be a list parsed by argparse in the form:
    [...] -mgm Fe 5 Nb 0.6 O 0.6 [...]
    which is then converted to a dictionary:
    d = {
        'Fe': 5.,
        'Nb': 0.6,
        'O': 0.6
        }
    """
    if magmoms is None:   return atoms
    else: print("Magnetic moments are set with ASE: %s"%magmoms)

    elements = magmoms[::2]
    values = magmoms[1::2]
    d = dict(zip(elements, values))
    init_mgm = []
    for s in atoms.symbols:
        if s not in elements:
            init_mgm.append(0)
        else:
            init_mgm.append(d[s])
    atoms.set_initial_magnetic_moments(init_mgm)
    return atoms

def set_hubU(atoms,hubU):
    if len(hubU)>0: #is not None:
        atoms.calc.set(ldau=".True.",ldauprint=0,ldautype=2,lmaxmix=6)
        ats=sorted(hubU.keys())
        ldauu=[];ldaul=[]
        #asyms=np.unique(atoms.get_chemical_symbols())#this automatically sorts the atom order
        asyms = list(dict.fromkeys(atoms.get_chemical_symbols()))
        #print (atoms.get_chemical_symbols(),asyms)
        for at in asyms:
            if at in ats: #continue
                if len(hubU[at].keys())>1: print("WARNING: cannot set multiple U values for the same atom type (%s), setting to 0... "%at);ldauu.append(0);ldaul.append(0);continue
                orb="".join(hubU[at].keys())#[0] indexing in dict_key format is a problem in Python3.
                ldauu.append(float(hubU[at][orb]))
                if orb=="p":ldaul.append(1)
                elif orb=="d":ldaul.append(2)
                elif orb=="f":ldaul.append(3)
            else:ldaul.append(0);ldauu.append(0)
        atoms.calc.set(ldauj=[ 0 for i in range(len(asyms)) ])
        atoms.calc.set(ldauu=ldauu,ldaul=ldaul)
        atoms.calc.set(lasph=1)#This is essential for accurate total energies and band structure calculations for f-elements (e.g. ceria), all 3d-elements (transition metal oxides), and magnetic atoms in the 2nd row (B-F atom), in particular if LDA+U or hybrid functionals or meta-GGAs are used, since these functionals often result in aspherical charge densities.
    return atoms


def Popen4(cmd):
    proc=Popen(cmd,shell=True, stdin=PIPE, stdout=PIPE, close_fds=True)
    out,err=proc.communicate()
    #out=proc.stdout.readline() #does not work.

    if version_info[0] < 3: out2=[x for x in str(out).replace("b'",'').replace("'",'').split("\n") if x!=""] #python3 adds \\n to the end, python2 adds \n
    else:out2=[x for x in str(out).replace("b'",'').replace("'",'').split("\\n") if x!=""]

    return out2,err

def vasp_continue(ifDel=0):
        fl = listdir(".")
        fl.sort()
        flag=0
        ind=0
        for fn in fl:
                if fn[0:3]=="RUN" :
                        ind = int(fn[3:])

        run="RUN%d"%(ind+1)
        print ("Previous data stored in %s."%run)
        system('mkdir %s'%run)
        system('cp * %s/ 2> /dev/null '%run)
        system('cp CONTCAR POSCAR') #this was missing in the previous version !!
        if ifDel: system('rm WAVECAR CHGCAR')

def call_vasp_v2(fname='',exe=None,xc='pbe',mgms=None,hubU={}): #atoms=None,

    if not exe: exe=getenv('VASP_COMMAND') ; 
    if not exe: exe='vasp_std'
    
    os.environ['VASP_COMMAND']=exe
    print("VASP command: ", exe) #, os.environ['VASP_COMMAND'])

    cwd=os.getcwd()

    seed=fname.split('.')[0]
    try:chdir(seed)
    except:print('Cannot change to %s directory, using %s instead'%(seed,cwd))

    print('Working dir: %s'%getcwd());stdout.flush();

    if args.makepotcar: make_potcar(xc=args.potcarxc,wDir='.')

    flag=0 #whether to start a new/continuation run 
    try:
        calc = Vasp(restart=True)
        atoms = calc.get_atoms()
        print ("\nVASP run was read succesfully from OUTCAR.")
        if Vasp.read_convergence(calc): print('Geom opt already converged...')
        else:
            print('Geom opt not converged; running a continuation job...')
            flag=1

    except:
        print ("VASP run could not be read, starting a new run...")
        flag=1

    if flag:
        calc=Vasp()
        calc.read_incar(filename='INCAR')
        if os.path.exists("./OUTCAR"):   vasp_continue()
        atoms=ase.io.read("POSCAR",format="vasp")
        #atoms=ase.io.read(fname)
        #calc.set(xc="pbe",ibrion=2,setups='recommended')#TODO: POTCAR: add others or take from the user
        calc.set(xc=xc)#,ibrion=2)
        calc.directory="."#cdir
        setups='recommended'
        #setups='minimal'
        calc.set(setups=setups)
        atoms.set_calculator(calc)

    # Adding MAGMOM and ISPIN to INCAR if -mgm or --magmoms is defined.
    atoms = handle_magmoms(atoms=atoms, magmoms=mgms) 
    if len(hubU)>0: atoms=set_hubU(atoms,hubU)

    Ep = atoms.get_potential_energy() 
    chdir(cwd)

    return Ep,atoms

def hashid(n=6):
    return ''.join([random.choice(string.ascii_lowercase + string.digits) for n in range(n)])


def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def get_potcar(elements, xc):
    ppdir = os.path.join(vasppp, xc)

    pp = ''
    for e in elements:
        #print os.path.join(ppdir, e, 'POTCAR')
        pp += open(os.path.join(ppdir, e, 'POTCAR')).read()

    return pp

def make_potcar(xc="potpaw_PBE",elements=None,wDir='./'):
    try:    vasppp = os.environ['VASP_PP_PATH']
    except: print ("VASP_PP_PATH is not defined, but neededor VASP calculation setup. Terminating. ");exit()

    print('Making potcar using %s'%vasppp);stdout.flush();


    if elements==None:#Try to read it from POSCAR if not explicitly given.
        try: #VASP5 format.
            elements = open(wDir+"/POSCAR").readlines()[5].strip().split()
        except:
            print ('Elements not given on command line and POSCAR not found')
            exit(-1)

    pp = get_potcar(elements, xc)

    p = open(wDir+"/POTCAR", 'w')
    p.write(pp);    p.close()

def find_interstices(atoms,tol=1e-6,sg=None,verb=None):#,prim=None):
    #TODO: only compatible with convetional cells (not with primitive cells). Easiest solution: first add the interstitials and then find the primitive cell. May not work with the repeated supercells (even with convetional ones), start rom standard cells

    from fractions import Fraction
    from ase.spacegroup import Spacegroup,get_spacegroup

    #get spacegroup
    if sg:        SG=Spacegroup(sg)
    else:        SG=get_spacegroup(atoms, symprec=tol)
    if verb: print ("Original space group: ",SG.no,SG.symbol)

    scpos=atoms.get_scaled_positions()
    if verb: print('Original scaled/fractional positions: \n',scpos)   

    if verb:print ('\nReading wyckoff.dat...')
    data={}
    with open(environ['HOME']+"/wyckoff.dat",'r') as file:
        flag=0
        i=0
        tmp={}
        for ln in file:
            if '#' in ln[0:1]: continue
            i+=1
            x=ln.split()
            if ln.strip()=='':
                if int(sg.split('-')[0]) != SG.no: tmp={};continue #only store the relevant space group.
                data[sg]=tmp;tmp={};flag=0; continue; #print (sg,data[sg]);continue
            elif len(x)==3 and ln[0] !='(' :wyck='%s%s'%(x[0],x[1]);tmp[wyck]=[]
            elif len(x)>5 and ln[0] !='(' : sg='%s-%s'%(x[0],x[1]) ; 
            elif ln[0]=='(': 
                for crd in x: #TODO: Double check if the x,y,z and addition/substraction of fractions are readand handled correctly (when readin from wyckoff.dat).
                    crd_bkp=deepcopy(crd)
                    crd=crd.strip('(');crd=crd.strip(')')
                    crd=crd.replace('-x','');crd=crd.replace('-y','');crd=crd.replace('-z','')
                    crd=crd.replace('+x','');crd=crd.replace('+y','');crd=crd.replace('+z','')
                    crd=crd.replace('x','');crd=crd.replace('y','');crd=crd.replace('z','')
                    #tmp[wyck].append([float(a) if ['x','y','z'] in a  else 1 for a in crd.split(',')  ])
                    try:tmp[wyck].append([float(Fraction(a)) if a != ''  else 0. for a in crd.split(',')  ])
                    except: print ('Error! :  ', sg,wyck,crd_bkp,crd)


    data2={}
    for key in list(data.keys()): 
        data2[key]={}
        for w in list(data[key].keys()):
            newcrd=data[key][w][0]
            equiSites=SG.equivalent_sites(newcrd,onduplicates='replace')[0] #Equi site of the first site should cover the other sites as well.
            #print (w,equiSites)
            data2[key][w]=[]
            for crd in equiSites: #SG.equivalent_sites([data[key][w][0]])[0]:
                #if crd not in scpos: #This one is supposed to work but somehow misses some obvious matches (due to the difference being larger than the default value).. Therefore switched to allclose comparison for each array element.
                flag=0
                for sc in scpos:
                    if np.allclose(sc, crd):flag=1;break 
                if not flag:
                    #print('%s,%s: %s not in atoms... %s '%(key,w,crd, equiSites))
                    data2[key][w].append(crd)

    a,b,c= atoms.get_cell()
    for i,key in enumerate(sorted_nicely(list(data2.keys()))):
        if i==1:break #Take only the first setting
        if verb: print(key)
        sites=[];    Va_sites=[]
        for w in sorted_nicely(list(data2[key].keys())):
            if len(data2[key][w])!=0: 
                if verb: print (w,[" ".join(['%.3f' %y for y in x])  for x in data2[key][w]])
                #for val in data2[key].values():
                sites.extend(data2[key][w])   

        try:ulist=np.unique(np.unique(sites,axis=0),axis=0)
        except:ulist=np.array([])

        flag=0 #Sometimes np.unique fails to detect equal arrays/sites, so this is a fail-safe approach to delete repeating sites. 
        for crd in ulist:
            flag=0
            for u in Va_sites: 
              if np.allclose(crd,u):flag=1;break 
            if flag:continue
            r=crd[0]*a+crd[1]*b+crd[2]*c #fractional to Cartesian conversion, works!!
            atoms.append(Atom('X',r)) 
            Va_sites.append(crd)

    if verb: print ('\n%d unique vacancy sites detected.'%(len(Va_sites)))#, Va_sites))
    if verb: 
      for x in Va_sites:print (x)
    if verb: print (atoms)
    #view(atoms)
    try:SG=get_spacegroup(atoms, symprec=tol)
    except: Sg=None
    print(SG)
    print ("\nNew space group after adding all interstices: ",SG.no,SG.symbol)

    if verb: print('New scaled/fractional positions: \n',atoms.get_scaled_positions())

    #view(atoms)
    return atoms


def find_equisites(atoms,all_W,all_w,tol=1e-6):
    noAtoms=len(atoms)
    equiSites=np.zeros((noAtoms))
    cpos=atoms.get_scaled_positions() #use the scaled positions instead!!
    coords=np.zeros((4,1))
    done=[] #already matched atoms.
    flag=0
    for at,atom in enumerate(atoms): #coords of the current atom.
        if at in done: continue
        coords[0:3,0]=cpos[at][0:3]
        coords[3,0]=1.
        flag=0
        for i,W in enumerate(all_W):#Find symmetry equi sites and match them to other available atoms.
                trans=np.zeros((4,4))
                w=all_w[i]
                trans[0:3,0:3]=W[:,:]
                trans[0:3,3]=w[0:3]
                trans[3,:]=[0.,0.,0.,1.]
                new_coords=np.dot(trans,coords)
                #assure the coordinates are btw 0 and 1.
                #for j in range(len(conf)):#no of atoms in conf
                for k in range(3):
                    if np.isclose(new_coords[k],0.): new_coords[k] = 0. 
                    elif np.isclose(new_coords[k],1.): new_coords[k] = 0. #to make it comparable to initial ASE-type coords (as 1. is automatically set to 0. in ASE).
                    elif new_coords[k] <0: new_coords[k] +=1 
                    elif new_coords[k] >1: new_coords[k] -=1 
                #print ('Atom Id: #%d, symm op #%d, new coords: %s'%(at,i,new_coords.T))

                #now match it to other atoms.
                for j in range(noAtoms):
                    if atoms[j].symbol != atoms[at].symbol: continue
                    crd=cpos[j]#coords of other atom.
                    if np.dot((new_coords[0:3].T-crd[:]),(new_coords[0:3].T-crd[:]).T)< tol:
                        #print ("Matched: ",atoms[j],atoms[at])
                        equiSites[j]=int(at)
                        done.append(j)
                        flag=1#;break
                #if flag:break

    return [int(x) for x in equiSites]

def dist(arr1,arr2):
    #Distance btw two atoms. Works with two 1x3 arrays.
    if len(arr1)==4: arr1=arr1[1:4]
    if len(arr2)==4: arr2=arr2[1:4]
    return math.sqrt((arr1[0]-arr2[0])**2+(arr1[1]-arr2[1])**2+(arr1[2]-arr2[2])**2)

#@jit(nopython=True, nogil=False,parallel=False)
def minDist(arr1,arr2,latt):
    #Finds the minimum distance from the images applying the PBC. (Minimum image convention).
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

def volume(cell):
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
            angle = 180.0 / np.pi * np.arccos(x)
        else:
            angle = 90.0
        angles.append(angle)
    if radians:
        angles = [angle * np.pi / 180 for angle in angles]
    return np.array(lengths + angles)

def lenCell_old(cell):
    res=[]
    for i in range(len(cell)):
        summ=0.0
        for j in range(3):
            summ += cell[i][j]**2
        res.append(math.sqrt(summ))
    return res

def compCells(cell1,cell2):#Compare cells.
    len1=lenCell(cell1);    len2=lenCell(cell2)
    #print len1,len2,tuple([int(round(len1[i]/len2[i])) for i in range(3)])
    return tuple([int(round(len1[i]/len2[i])) for i in range(3)])

def cartesian(arrays, out=None): #Not required anymore.
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def combinations2(iterable, r): #from itertools
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(list(range(r))):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)

#import operator as op
def ncr(n, r): #gives combination.
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, list(range(n, n-r, -1)))
    denom = reduce(op.mul, list(range(1, r+1)))
    return numer//denom


def sort_dict_by_val(data,index=0):
    #index: index to use if values of the dictionary is not scalar.
    keys=list(data.keys())
    if len(keys)==1: return keys

    skeys=[keys[0]]#sorted keys
    for i in range(len(keys)):
        key=keys[i]
        val=data[key]
        try: val=val[index]
        except:None
        flag=0
        for j in range(len(skeys)):
            key2=skeys[j]
            val2=data[key2]
            try: val2=val2[index]
            except:None
            if val <= val2 and i!=0: skeys.insert(j,key);flag=1;break
        if not flag and key not in skeys:skeys.append(key)#not to miss the largest element.

    return skeys


#Fixde version !!
def boltz_dist(energies,T=298.15,omega=[],verb=0):#Return the occupation probabilities of the configurations at a given temperature based on their energies.
    #Higher degenracy for an irreducible configration means higher config entropy, and thus lower reduced energy. (config entropy lowers the relative enregy.)
    kb= 8.6173303*10**-5 #Boltzmann constant (eV/K).
    if len(omega)==0:#If the degeneracies are not given explciitly, all set to 1.
        omega=[1 for E in energies]

    if 1: #Get relative energies.  Doesn't really matter as long as you use the normalised factors.
        mn=min(energies)
        rel_energies=[i-mn for i in energies]
    probs=[]
    for E in rel_energies:
        try:probs.append(math.exp(-E/kb/T))
        except:probs.append(0.0)
    #Normalise    
    Z=sum(probs) #i.e. partition fnc
    probs=[Pn/Z for Pn in probs]
    #print (probs)

    #Configurational statistics as given in R. Grau-Crespo et al. J.Phys: COndens. Matter, 19,2007,256201 and DOI:10.13140/RG.2.1.3561.4161 (Mol2Net 2015 paper)
    print ('\nConfigurational thermodynamics at T= %d K\n'%T)
    E_avg=sum([energies[i]*float(probs[i]) for i in range(len(energies))])
    rel_E_avg=sum([rel_energies[i]*float(probs[i]) for i in range(len(energies))])
    if verb: print ("Average energy of the sytem in configurational equilibirum,  E=%.5f eV"%E_avg)

    #F=-kb*T*np.log(Z)
    F=-kb*T*sum([np.exp(-E/kb/T) for E in rel_energies])
    if verb: print ("Configurational free energy in the complete space, F=%.5f eV"%F)

    S= (rel_E_avg-F)/T
    if verb: print ("Configurational entropy in the complete space, S=%.5f eV/K"%S)

    #Smax=kb*np.log(len(energies)) #not right
    Smax=kb*np.log(sum(degens)) #SHould be total no of possible configs (so  sum(degens))
    if verb: print ("Upper limit of config. entropy, Smax= %.5f eV/K"%Smax)

    #Now count in the degenaricies of the configs.
    Sm=[kb*T*np.log(om) for om in omega] #degeneracy entropy
    if verb: print ("Degeneracy entropy, Sm= %s"%(['%.3f'%m for m in Sm])) 

    #for i,E in enumerate(energies):
    Em_bar=[energies[i]-Sm[i] for i in range(len(energies))] #Reduced energy #there is a typo in the Grau-Crespo, J. Phys.: Condens. Matter 19 (2007) 256201 paper, Eqn 12: T should not be there.
    if 1: 
        mn=min(Em_bar)
        rel_Em_bar=[i-mn for i in Em_bar]
    #print ('Reduced energies, Em_bar= %s'%(['%.5f '%m for m in Em_bar ])) 

    Pm=[np.exp(-rel_Em_bar[i]/kb/T) for i in range(len(energies))] 
    Z_bar=sum(Pm)
    #Pm_bar=[(1/Z_bar)*np.exp(-rel_Em_bar[i]/kb/T) for i in range(len(energies))] #reduced  probability for an independent config.
    Pm_bar=[P/Z_bar for P in Pm] #reduced  probability for an independent config.

    E_avg=sum([rel_Em_bar[i]*Pm_bar[i] for i in range(len(energies))])

    #F=-kb*T*np.log(Z_bar)
    F=-kb*T*sum([np.exp(-E/kb/T) for E in rel_Em_bar])
    if verb: print ("Configurational free energy in the reduced config. space, F=%.5f eV"%F)

    S= (E_avg-F)/T  #Check this !!
    if verb: print ("Configurational entropy in the reduced config. space, S=%.5f eV/K"%S)

    if verb: print ("Original probabilties for the configurations: %s"%(['%.3f'%m for m in probs ]))
    if verb: print ("Reduced probabilties for  independent configurations: %s"%(['%.3f'%m for m in Pm_bar ]))

    return Pm_bar,Em_bar#,F #,E_avg


if __name__ == '__main__':

    #Common variables.
    inpdir="INPUTS"
    seed='config'
    outdir="OUT-FILES"    

    try:    
        vasppp = os.environ['VASP_PP_PATH']
        ppdirs=os.listdir(vasppp)
        ppdirs.sort()
    except: ppdirs=""


    parser = argparse.ArgumentParser(description='This script generates the input structures required for a configuration enumeration (including vacancies and dopants) for any software supported by ASE. \nScript can also create the input files (prim.json and config_list.json) required for CASM for further analysis. \nRequires ASE and SPGlib packages.\n Sample usage: config_enum.py -i LLZO-Ia-3dCollCode261302.cif -prim -at Li -wl 96h  -dt Va -nd 0:2  --casm -sf -ot vasp -o POS')

    parser.add_argument('-i', '--inpf', type=str,required=True,nargs='*',help='Input file(s), any file type compatible with ASE.')
    parser.add_argument('-o','--outf', type=str,required=False, help='Output file name. Def: Automatic naming: e.g. %s_ID.'%seed)
    parser.add_argument('-it','--itype', type=str,required=False, help='Input file type. Def: determined automatically from the extension.')
    parser.add_argument('-ot','--otype', default='res',type=str,required=False, help='Output file type, default: .res (SHELX format)')
    parser.add_argument('-od','--odir', default=outdir,type=str,required=False, help='Output directory, default: %s'%outdir)

    parser.add_argument('-ow','--overwrite', default=False,action='store_true', help='overwrite if output folder exists. Def: No')
    parser.add_argument('-rs','--restart', default=False,action='store_true', help='overwrite if output folder exists. Def: No')
    parser.add_argument("-iseed","--inpseed", type=str, default=None, help="Seed name for reading the .param and .cell files in order to use the settings within.")
    parser.add_argument('-sf','--sepfol', default=False,action='store_true', help='to save geometries in separate folders (e.g. for VASP runs). Default: all output is written in the same directory (%s).'%outdir)
 
    parser.add_argument('-prim','--ifPrim', default=False,action='store_true', help='use the primitive cell instead of supercell (requires SPGlib installed). Highly recommended for computational efficiency. Def: false')
    parser.add_argument('-ref', '--initCrysFile', type=str,help='File for obtaining the initial crystal site (symmetry) information. Can be any type that ASE supports. Def: same as the input file (-i)')
    parser.add_argument('-tol', '--tol',type=float, default=1e-3,help="The symmetry tolerance. Def: 1e-3")
    parser.add_argument('-nosymm','--nosymm', default=False,action='store_true', help='switch off the use of symmetry info, e.g. equivalent sites. (to remove SPGlib dependancy; not recommended).')
    parser.add_argument('-r','--rep', nargs=3, type=int,help="Repeat the input structure in a, b, c directions to get a supercell, e.g. -r 2 2 1")

    parser.add_argument('-n', '--atoms_list', nargs='*', type=int,help='list of atom ids representing the crys sites where the vacancy/dopant will be placed during enumeration (count starts at 1). Example usage: -n 1 3 4 ')
    parser.add_argument('-nl', '--list',help='atom list(count starts at 1); e.g. -nl 1:128')
    parser.add_argument('-at', '--aType',nargs='*', type=str,help='atom type to be included in enumeration; e.g. -at Na Li')
    parser.add_argument('-wl', '--wycklist',nargs='*',type=str,help='List of Wyckoff positions of crys. sites to include in enumeration. A list of available sites can be obtained via dry run (-dry); e.g. -wl 24d 12a')
    parser.add_argument('-excl', '--excludelist',nargs='*',type=str,help='List of crystal sites to exclude from the configuration eneumeration (these atoms will not be considered even though selected by -n, -nl,-at or -wl earlier).  e.g. 1:4 24  for [1,2,3,4,24], atom numbering starts at 1.')
    parser.add_argument('-dexcl','--dopeexcluded', default=False,action='store_true', help='Add dopant/vacancy to the atoms/crys-sites given in the xcluded list. This option is good for doing iterative vacncy/dopant addition with the previous knowledge of lower dopant/vacancy compositions. Def: False.')
    parser.add_argument('-dt', '--dType', required=True, type=str,help='Dopant type to consider in the enumeration; This can be an atom symbol (e.g. Na) or vacancy (e.g. Va)')
    parser.add_argument('-dexcl_type', '--dexcl_type', required=False, type=str,help='Dopant type to consider for the sites given in the excluded list (see -dexcl keyword for details). Def: equal to dType', default="")
    parser.add_argument('-nd', '--noDopants',required=True,nargs='*',type=str,help='(List of) number of dopant/vacancy to consider in enumeration; e.g. 0:4 24  for 0,1,2,3,4 and 24 vacancies)')

    parser.add_argument('-mod','--modify', type=str,help="To make some final touches on the configurations selected for enumeration. Keep in mind that these final modifications shoudl refer to the new atom indices. You can combine multiple modifications using multiple 'statements' (enclosed by single-quotations), the order of the statements will be strictly followed. This is a useful option to do necessary modifications to compansate for the dopant effect  e.g. balancing the total system charge  by deleting additional (nearest) cations upon introducing higher-valence cation(s). Def: No additional modifications.  Example usage: -mod 'replace all within 2 of Al with Va'.")
    parser.add_argument('-ai','--addint', default=False,action='store_true', help='To add the interstices as X atoms based on the  Wyckoff positions data for the reference symmetry gorup. Default: False.')
    parser.add_argument('-ase','--ASE_symm', default=False,action='store_true', help='Verbose output. Default: False.')

    parser.add_argument('-castep','--castep', default=False,action='store_true', help='Run quick, low-accuracy CASTEP calculations to estimate the configration energies. Default: Only Lennard-Jones and Morse potentials (as in ASE) are used.')
    parser.add_argument('-vasp','--vasp', default=False,action='store_true', help='Use VASP to compute configration energies. Inputs are read from the args.idir folder')
    parser.add_argument('-vexe','--vaspexe', type=str,required=False, default='vasp_std',help='Vasp exectuable. Def: vasp_std')
    parser.add_argument('-cexe','--castepexe', type=str,required=False, default='castep.mpi',help='Vasp exectuable. Def: castep.mpi')
    parser.add_argument('-casm','--casm', default=False,action='store_true', help='to create prim.json and config_list.json files for CASM. Default: False.')

    parser.add_argument('-copy','--copyinp', default=False,action='store_true', help='to copy the files contained in "%s" folder. Default: False.'%inpdir)
    parser.add_argument('-id','--idir', default=inpdir,type=str,required=False, help='Input directory to copy the files from, default: %s'%inpdir)

    parser.add_argument('-mp','-makepotcar','--makepotcar', default=False,action='store_true', help='to compile POTCAR (for VASP) using actual atomic content. The environment variable ($VASP_PP_PATH) should be defined beforehand. Default: False.')
    parser.add_argument('--potcarxc', default="potpaw_PBE",type=str,choices=ppdirs,help='XC functional to use in making of POTCAR. Def: potpaw_PBE')
    parser.add_argument('-xc','--xc', type=str,required=False, default='pbe',help='Exchange correlation functional to use (pseudopots will be selected accordingly, e.g. LDA, PBE,PBEsol, etc. Def: PBE')
    parser.add_argument("-hubU", "--hubU", type=str,nargs="*",help="For defining the Hubbard U parameters (in eV) for specific atom and orbital types, e.g. -hubU Fe d 2.7, Sm f 6.1")
    parser.add_argument("-mgm","--magmoms", default=None, nargs="*", required=False, help="Magnetic moments for a collinear calculation. Eg, 'Fe 5.0 Nb 0.6 O 0.6' Def.: None. If a defined element is not present in the POSCAR, no MAGMOM will be set for it.")

    parser.add_argument('-np', '--nprocs',type=int, default=32,help="No of processes to start for each CASTEP/VASP calculation through srun/mpirun. Def:32")
    parser.add_argument('-nt', '--ntasks',type=int, default=1,help="No of CASTEP/VASP tasks to run simultaneously. Def:1")
    parser.add_argument('-nn', '--nnodes',type=int, default=1,help="No of nodes to run CASTEP/VASP runs through srun. Def:1")
    parser.add_argument('-mpi','--mpirun', default=False,action='store_true', help='Use mpirun for parallel runs. Default: srun is used')

    parser.add_argument("-t","--temp", type=float, default=298.15, help="Temperature at which the configrational thermodnamics is computed. Def: 298.15K")

    parser.add_argument("-seq","--seq", type=int, default=0, help="If to run an automated sequential doping to reach the target number of dopants (), based on configuration energies/symmetry as selection criterion (--seqtype) at each step. Def: No sequential doping.")
    parser.add_argument("-spick","--seqpick", type=int, default=1, help="How many configs to pick at each step in a sequential doping. Def: 1")
    parser.add_argument("-stype","--seqtype", type=str, choices=['E','S','D','R'],default='E', help="How to sort the configurations at each step of  a sequential doping. Options: 'E': Based on Energy (low-energy configs are picked); 'S': based on symmetry (highest-symmetry configs picked); 'D': config(s) with highest degenaracy are picked; 'R': configs are picked randomly. Def: 'S' ")

    parser.add_argument('-v','--verb', default=False,action='store_true', help='Verbose output. Default: False.')
    parser.add_argument('-dry','--dry', default=False,action='store_true', help='Dry run to get the equivalent (Wyckoff) positons. Default: False.')
    parser.add_argument('-noskip','--noskip', default=False,action='store_true', help='Do not skip problematic configs (and assigning them with 0 energy and P1 symmetry). Default: Bad configs are skipped.')

    args = parser.parse_args()
    initT0=time.time()

    print()


    outdir=args.odir
    inpdir=args.idir
    inpseed=args.inpseed

    if args.castep:args.seqtype='E'

    if args.seq==0:args.seq=int(args.noDopants[0])

    #Parse the Hubbard_U info and make the dict
    hubU={}
    if args.hubU:
        U=" ".join(args.hubU)
        xxx=U.split(',')
        for xx in xxx:
            x=xx.split()
            if not x[0] in hubU:  hubU[x[0]]={x[1]:x[2]}
            else: hubU[x[0]][x[1]]=x[2]
        print ("Hubbard U values to be used: ",hubU)

    #Get the  dopant/vacancy number list.
    nDops=[]
    for y in args.noDopants:
        x=y.split(':')

        if len(x)>1:
            nDops.extend(list(range(int(x[0]),int(x[1])+1)))
        else:
            nDops.append(int(x[0]))
    nDops.sort()



    if args.seq!=0 :
        if len(nDops)>1: print ('Multiple no of dopant entries (-nd) is not supported with -seq option...');exit()

        #args.inpf = '../' + args.inpf
        try:args.initCrysFile = '../'+args.initCrysFile
        except:
            if len(args.inpf)==1: args.initCrysFile = '../'+args.inpf[0]
            else:  print('A reference structure file must be given using --ref tag to extract the crystal sites and symmetry operations');exit() 
        seq=args.seq
        if os.path.exists('Step1'):
            if args.overwrite: 
                print("StepX folders exist. Overwriting as requested.")
                os.system('rm -rf Step* ')
            elif args.restart: 
                print("StepX folders exist. Will try to restart the VASP calculations as requested.")
            else: 
                print("StepX folders exists. Not overwriting, use -ow to overwrite.")
                exit()
    else:
        seq=nDops[0]

    #Start the sequential doping loop here
    step=0;tDop=deepcopy(nDops[0])
    for cDop in range(0,tDop,seq):
        step+=1
        if args.seq>0: system('mkdir -p Step%d'%(step)) ; os.chdir('./Step%d'%(step))  ; print ('Starting Step %d...'%(step));    system('rm -rf Picked; mkdir -p Picked')

        if (tDop-cDop) <seq: nDops[0]=tDop-cDop
        else:nDops[0]=seq

        degen_outf=open('degeneracies','w')
        if args.casm: args.sepfol=True #CASM needs separate folder for each config.
        if args.copyinp: 
            if not os.path.exists (inpdir) or len(os.listdir(inpdir))==0:
                print("Copying input files is requested, but the %s directory is empty. Switching off -copy."%inpdir)
                args.copyinp=False
        initT=time.time()


        if step>1:args.inpf=picked;str1=''
        else:        str1=" ".join(argv)+'\n\n'
        picked=[];seqconfigs=[ [],[],[],[] ] #all configs for the current step from different inpf.

        #Start input file loop
        for find,inpf in enumerate(args.inpf):
            if args.seq>0:inpf = '../' + inpf
            print('Working on %s...'%inpf)
            fname=inpf.split('/')[-1].split('.')[0]
            outdir='input_%d'%(find) #outdir=fname   #os.system('mkdir -p ./%s'%(inpf));os.chdir('./%s'%(inpf))

            if os.path.exists(outdir):
                if args.overwrite: 
                        print("%s folder exists. Overwriting as requested."%outdir)
                        #os.system('rm -r ./%s'%(outdir))
                elif args.restart:
                        print("%s folder exists. Restarting as requested."%outdir)
                else: 
                        print("%s folder exists. Not overwriting, use -ow to overwrite."%outdir)
                        exit()

            os.system('mkdir -p ./%s'%(outdir))


            try:
                if args.itype: atoms=ase.io.read(inpf,format=args.itype)
                else: atoms=ase.io.read(inpf) #ASE automatically determines from the extension.
            except:
                print("Error in reading the input file (%s)"%inpf);    exit()

            #Outfile extension.
            ext=args.otype.split("-")[-1]
            if args.outf: outf=args.outf+"."+ext
            else: outf=inpf.split('/')[-1].split(".")[0]+"."+ext

            if 'X' in args.aType and args.addint: #Find the potential interstices/vacancies, checkign the Wykoff site list for the space group. This must be done before finding the primitive and repetitions
                if args.initCrysFile: #if an input file for crsy. site info is provided.
                    #Use an external ref. for symm ops. to determine the Wyckoff pos. Another option is to delete the disordered sites, works  for finding  a high symmetry (not always, so switched off now).
                    print ('%s is used as reference for the symmetry information.'%args.initCrysFile)
                    if args.initCrysFile.split("/")[-1]=="POSCAR": fm="vasp";args.tol=1e0
                    else: fm=None
                    atoms_crys=ase.io.read(args.initCrysFile,format=fm)   
                else: atoms_crys=deepcopy(atoms) #use input file as the template for symmetry.

                scaled_positions= atoms_crys.get_scaled_positions()
                cell=(atoms_crys.cell, scaled_positions, atoms_crys.numbers)
                sg=get_spacegroup(cell, symprec=args.tol)
                print (sg,int(sg.split('(')[-1].replace(')','')))
                atoms=find_interstices(atoms,tol=args.tol,verb=args.verb,sg=int(sg.split('(')[-1].replace(')','')))#input the space group no.        #prim=args.ifPrim,



            rpt_no=1
            if step==1 and args.ifPrim: 
                print("Finding the primitive cell using SPGlib as requested.")
                atoms_orig=atoms.copy()
                scaled_positions= atoms.get_scaled_positions()
                cell=(atoms.cell, scaled_positions, atoms.numbers)
                sg=get_spacegroup(cell, symprec=args.tol)
                print("Space group= %s with tol=%.2e"%(sg,args.tol))
                lattice, scaled_positions, numbers = find_primitive(cell, symprec=args.tol)
                atoms=Atoms(symbols=numbers,cell=lattice,scaled_positions=scaled_positions,pbc=True)

                #use lengths for comparison
                rpt=compCells(atoms.cell,atoms_orig.cell)
                rpt_no=rpt[0]*rpt[1]*rpt[2]
                if rpt_no==1:#use volumes for cell comparison. If cell lengths failed for a reason.
                    rpt_no=round(volume(atoms_orig.cell)/volume(atoms.cell))  

            if step==1 and args.rep: 
                atoms=atoms.repeat(args.rep)
                atoms=sort(atoms)


            if not args.nosymm:

                if args.initCrysFile: #if an input file for crsy. site info is provided.
                    #Use an external ref. for symm ops. to determine the Wyckoff pos. Another option is to delete the disordered sites, works  for finding  a high symmetry (not always, so switched off now).
                    print ('%s is used as reference for the symmetry information.'%args.initCrysFile)
                    if args.initCrysFile.split("/")[-1]=="POSCAR": fm="vasp";args.tol=1e0
                    else: fm=None
                    atoms_crys=ase.io.read(args.initCrysFile,format=fm)   
                else:atoms_crys=deepcopy(atoms) #use input file as the template for symmetry.

                if 0 and args.aType:        #Supercell code finds the symm operations by discluding the disordered sites (i.e. site for doping). Actually helps find the symmetry operations, even though some of the disordered sites were deleted.
                    del atoms_crys[[atom.index for atom in atoms_crys if atom.symbol in args.aType ]]#

                    #TODO: Assign the equivalent sites for the original system (atoms) using the symm ops from the atoms_crys (dwith deleted sites for higher symm). NEEDED for correct assignment of the wyckoff sites in case they are deleted.


                cell=(atoms_crys.cell, atoms_crys.get_scaled_positions(), atoms_crys.numbers)

                data = get_symmetry_dataset(cell, symprec=args.tol, angle_tolerance=-1.0, hall_number=0)

                print ("%d symmetry operations were detected. "%( len(data['translations'])))
                sg=get_spacegroup(cell, symprec=args.tol)
                print("Space group= %s with tol=%.2e"%(sg,args.tol))

                all_W=data['rotations']
                all_w=data['translations']

                if args.verb: print (data['equivalent_atoms'])
                if 0: #my own version also works well, and can be used.
                    data['equivalent_atoms']=find_equisites(atoms,all_W,all_w,tol=args.tol)


                #Determine the equivalent atoms and sites.
                equiSites={}    
                if data is not None:
                    for i in range(len(data['equivalent_atoms'])):
                        ea=str(data['equivalent_atoms'][i])

                        if ea not in equiSites: 
                            #This try, except needed when deleting disordered sites for getting higher symmetry, as the wyckoff positions for the disordered sites are missing.: TODO: find a way to get the wyckoffs for the missing sites w/out using SPGlib.
                            try:equiSites[ea]=[data['wyckoffs'][i],i]
                            except:equiSites[ea]=['X',i] 
                        else: equiSites[ea].append(i)

                    #Correct the keys.
                    keys=list(equiSites.keys())
                    keys.sort()
                    eS=deepcopy(equiSites)
                    equiSites={}
                    for i in range(len(keys)):
                        key = keys[i]
                        equiSites['equi_site%d'%(i+1)]=eS[key]
                else: 
                    print("No equivalent positions. All selected atoms will be treated separately.")
                    equiSites={}
                    for i in range(len(atoms_list)):
                        equiSites['equi_site%d'%(i+1)]=['a',i]

                    all_W=np.matrix([[1,0,0],[0,1,0],[0,0,1]])
                    all_w=np.matrix([[0,0,0]]).T    


                eSites={}
                keys=list(equiSites.keys())
                keys.sort()
                wyckoffs={}
                for key in keys: #loop over equi sites.
                    val=equiSites[key]
                    site='%d%s'%(rpt_no*(len(val)-1),val[0])
                    #nval=[]
                    if site in eSites:
                        eSites[site].extend(val[1:])
                    else: eSites[site]=val[1:]

                    #Other version from vasp_plot_RMS-traj.py
                    #if '%d%s'%(len(val)-1,val[0]) not in eSites: eSites['%d%s'%(len(val)-1,val[0])]=val[1:]
                    #else: eSites['%d%s'%(len(val)-1,val[0])].extend(val[1:])

                for key in list(eSites.keys()): #loop over wyckoff groups
                    val=eSites[key]
                    x=re.split('([0-9]+)', key)
                    if re.search('[A-Z]+',str(val[0])):continue
                    if x[1] != rpt_no*len(val): #update the wyckoff group name with correct no of site.
                        #Find diff atom types in the group.
                        try:ats=np.array([ atoms[i].symbol for i in val])
                        except:
                            ats=[]
                            for i in val:
                                try:ats.append(atoms[i].symbol)
                                except:continue
                            ats=np.array(ats)
                        ua=len(np.unique(ats)) #no of unique atom types.
                        nkey='%d%s'%(rpt_no*len(val)/ua,x[-1])
                        del eSites[key]
                    else:    nkey=key
                    eSites[nkey]=['%s%d'%(ats[i],val[i]) for i in range(len(ats))]
                    for v in val: 
                        try:wyckoffs[str(v)]=nkey
                        except:None


                print("The following equivalent sites were extracted from %s: "%inpf)
                #print(eSites)
                for eS in sorted_nicely(list(eSites.keys())):
                    print ("%s: %s"%(eS,sorted_nicely(eSites[eS]))) #This is not used anywhere else in the code. Just  for informing the user.
                print()

            else: #The nosymm case/
                all_W=np.matrix([[1,0,0],[0,1,0],[0,0,1]])
                all_w=np.matrix([[0,0,0]]).T    


            if args.dry:
                    print("Dry run: No inputs were created."); exit()


            ##########
            #Parse the user input as to atom selection.
            if args.aType or args.wycklist: #Determine the atomic indices of the atoms with the desired type.
                atoms_list=[]
                chemsym=atoms.get_chemical_symbols()
                #print(chemsym)
                for i in range(len(chemsym)):
                    at=chemsym[i]
                    if at=='Va':at='X'
                    try:wt=wyckoffs[str(i)]
                    except: wt=''
                    if args.aType and args.wycklist:
                        if at in args.aType and wt in args.wycklist: atoms_list.append(i)
                    elif args.aType: 
                        if at in args.aType: atoms_list.append(i)
                    elif args.wycklist:
                        if wt in args.wycklist: atoms_list.append(i)
                if len(atoms_list)==0: print("No atoms were selected based on your input. Check the atom selection  options (i.e. -at, -wl,-nl -n). Terminating...");exit()
            elif args.atoms_list:
                atoms_list=[] #args.atoms_list
                for at in args.atoms_list: 
                    if at == 0: print("Wrong atom index (atom count starts at 1)");exit()
                    else: atoms_list.append(at-1) #atoms_list[i] = int(atoms_list[i])-1
            elif args.list:
                noAtoms=len(atoms)
                x=args.list.split(':')
                x0=int(x[0])
                if len(x[1])==0: x1=noAtoms
                else: x1=int(x[1])
                if x0==0 or x1>noAtoms:
                    print("Wrong atom index (atom count starts at 1)");exit()  #atoms_list = range(len(atoms))
                atoms_list=list(range(x0-1,x1))
            else:
                print("The list of atoms (and their corresponding crytalline sites) which will be replaced by vacancy or dopants must be given. Use -n, -nl, -at or -wl for this purpose. See help (-h) for furhter information.")
                exit()

            print("\n%d atoms selected."%len(atoms_list))
            print("List of atoms (index starts at 0) selected for the enumeration: ",atoms_list)   

            eList=[]
            if args.excludelist:
                #eList=[]
                for y in args.excludelist:
                        y=y.replace(","," ")
                        x=y.split(':')

                        if len(x)>1:
                                eList.extend(list(range(int(x[0]),int(x[1])+1)))
                        else:
                                eList.append(int(x[0]))

                eList=[x-1 for x in eList]
                eList.sort()

                print ("Excluded list: ",eList)
                for val in eList: 
                        try:atoms_list.remove(val)
                        except:None
                print("New list of selected %d atoms: "%len(atoms_list),atoms_list)
                if args.dopeexcluded:
                        print ("Atoms in the excluded list will not be doped/left vacant as requested.")
                        #Determine the dopant type for the excld list.
                        if args.dexcl_type=="":dexcl_type=args.dType
                        else:dexcl_type=args.dexcl_type

                        if eList==None:  
                                print ("Exclude list is empty, switching off -dexcl keyword."); args.dopeexcluded=False



            chemsym=atoms.get_chemical_symbols()
            atomSyms=[]
            for i in chemsym:
                    if i not in atomSyms:atomSyms.append(i)

            ########################################
            #Parse the --modify keyword arguments.
            mod_keys=[];mod_type='';mod_value=0;mod_target=None;mod_replwith=None
            if args.modify:
                mod_keys=args.modify.replace("'","").split()
                flag=0
                if mod_keys[0] not in ["remove","replace"]: flag=1
                if mod_keys[2] not in ["within","nearest"]: flag=1
                if mod_keys[4] not in ["of"]: flag=1
                #if mod_keys[-2] not in ["with"]: flag=1

                if flag: 
                        print ("wrong usage of -mod keyword.");exit()

                if mod_keys[0]=="replace" or mod_keys[0]=="remove":
                        if mod_keys[1]=="all":mod_repl=atomSyms
                        else:mod_repl=[mod_keys[1]]

                        mod_type=mod_keys[2] #within or nearest
                        mod_value=float(mod_keys[3])
                        mod_target=mod_keys[5]
                        if mod_keys[0]=="replace":
                                mod_replwith=mod_keys[-1]
                        elif mod_keys[0]=="remove":
                                None


            print("\nNow creating the modified cells, applying the desired type and number of dopant/vacancy on the selected atoms.")

            atoms_orig=atoms.copy()


            if args.casm:
                outff=open('config_list.json','w')
                outff2=open('prim.json','w')
                str11="""{
          "supercells" : {
            "SCEL1_1_1_1_0_0_0" : {
        """
                str2="""{
          "basis" : [
        """     
                pos=atoms.get_scaled_positions()
                for i in range(len(atoms)):
                    at=atoms[i]
                    str2+="""    {
              "coordinate" : [ %14.12f, %14.12f, %14.12f ],
        """%(pos[i][0],pos[i][1],pos[i][2])
                    if i in atoms_list:#for selected atoms.
                        str2+="""      "occupant_dof" : [ "%s", "%s" ]
            },
        """%(at.symbol,args.dType)  #UPDATE THIS FOR diff type for excluded list !!!
                    else:
                        str2+="""      "occupant_dof" : [ "%s" ]
            },
        """%(at.symbol)


                #symbols=[]
                #for x in atoms.get_chemical_symbols():
                #    if x not in symbols: symbols.append(x)

                symbols   = list(dict.fromkeys(atoms.get_chemical_symbols()))


                #Final part.
                str2=str2[0:-2]+'\n'

                str2+="""  ],
          "coordinate_mode" : "Fractional",
          "description" : "%s ",
          "lattice_vectors" : [
            %s,
            %s,
            %s
          ],
          "title" : "%s"
        } """%(' '.join(symbols),[x for x in atoms.cell[0]],[x for x in atoms.cell[1]],[x for x in atoms.cell[2]],''.join(symbols))


                outff2.writelines(str2)#;outff2.flush()

            ########
            # Now create the defect configs here
            #########
            cnt=0 #(current id/no of iredducible config/structures generated.

            symbols   = list(dict.fromkeys(atoms.get_chemical_symbols()))
            #symbols = atoms.get_chemical_symbols()
            #print(symbols)
            for nD in nDops:
                if nD > len(atoms_list):break
                nc=ncr(len(atoms_list),nD)
                if nc>2e4: #Too many possibilities.
                    print("Too many combinations (%.1e) to handle for %d vacancies/dopants, skipping..."%(nc,nD)); #continue
                c=[x for x in combinations(atoms_list,r=nD)] #gives the selected atom ids
                #print("%d configuration(s) found for %d vacancies/dopants."%(len(c),nD))

                if len(args.aType)>1: #This is to ensure to include only the configs where all of the target sites are being doped (for instance, if you choose Li and Nb sites to dope, then the configs where both Li and Nb sites are doped).
                    cnew=[]
                    syms=atoms.get_chemical_symbols()
                    for x in c:
                        if len(np.unique([syms[i] for i in x ])) == len(args.aType):  cnew.append(x)
                    #print("Reduced  %d configuration(s) found for %d vacancies/dopants."%(len(cnew),nD))
                    c=cnew

                print("%d configuration(s) found for %d vacancies/dopants."%(len(c),nD))

                #########
                #Now check the symm equivalence of the configs
                ######
                all_eSites=[] #irreducible configs saved so far
                #all_eSites=np.array([]) #irreducible configs saved so far
                degens=np.zeros((len(c))) #degeneracy of each config
                all_configs=[];all_files=[]
                scpos=atoms.get_scaled_positions(wrap=True)
                scpos=np.where(np.isclose(scpos, 1.0), 0.0,scpos) #Replace all 1.0 entries with 0.0 for compatibility.
                initT1=time.time()
                for cind,conf in enumerate(c): #iterate over configurations. #combinations(atoms_list,r=i):
                    if cind%100==0 or cind+1==len(c): 
                        print ("%.1f%% %d"%(float(cind+1)/len(c)*100,len(all_configs)),end=', ');stdout.flush();     print("time: %.2f sec."%( time.time()-initT1))


                    atoms=atoms_orig.copy()
                    conf=list(conf);conf_orig=deepcopy(conf);

                    if not args.nosymm:
                       #Now find the symetrically-equivalent configrations, by applying the symmops. Still iterating over the available configs. #Only include the selected ones for the given config/combination !!
                        coords=np.zeros((4,len(conf)))
                        eFlag=0
                        for i,at in enumerate(conf): #iterate over the atoms in the current combination
                            #cpos=atoms[at].position #use the scaled positions instead!!
                            cpos=scpos[at] #use the scaled positions instead!!
                            coords[0:3,i]=cpos[0:3] #4xnoDopants
                            coords[3,i]=1.



                        if cind >0: #original, faster for <=2 dopants
                            #all_eSites2=np.unique(all_eSites,axis=0)
                            for i,equi in enumerate(all_eSites): #i is the index of the actual/reduced config id ,cind is the original/running config id.
                                ifFound=np.zeros((len(conf)))
                                used=np.zeros((len(conf)))
                                for crd in equi: #loop over atoms in the equivalent configs This loop takes time.
                                    ifFound=np.zeros((len(conf)))
                                    used=np.zeros((len(conf)))

                                    #Also consider the equivalent cases where the atomic order has changed !
                                    for k in range(len(conf)): #This is the bottleneck, should be fully numpied.
                                        for j in range(len(conf)): #compare with other atoms .
                                            if used[j]:continue #same atom should not be used twice for mapping.
                                            #try:
                                            diff=np.dot((coords[:,k]-crd[:,j]).T,(coords[:,k]-crd[:,j]))
                                            #except:diff=np.dot((coords[:,k]-equi[:,j]).T,(coords[:,k]-equi[:,j]))
                                            if diff <= args.tol: #or abs(diff)-1.0 <=args.tol:  #much faster than np.allclose and catches the 0.0 and 1.0 coorditanets (essentially equivalent).
                                                if 1 and args.verb: print('Matched: ', cind, i, ['%.3f %.3f'%(float(coords[m]),float(crd[m])) for m in range(len(coords))])
                                                used[j]=1
                                                ifFound[k]=1;break
                                            elif 0 and args.verb: print('Not matched: ', cind, i, ['%.3f %.3f'%(float(coords[m]),float(crd[m])) for m in range(len(coords))])

                                    if np.sum(ifFound)==len(conf): #all atoms have been matched to a symm-equi one.
                                        eFlag=1
                                        degens[i]+=1; #print(degens)
                                        break

                                if eFlag:
                                    #if args.verb: print ('Matched configs: %d and %d' %(i,cind))
                                    break

                        if eFlag:continue  #if it's a symm-equivalent config, thenskip it.
                        elif args.verb: print('New config: ',cind)

                        #""" #Original one, fast and gives the owest amount of confgis.
                        #Find the set of symm-equivalent sites for the current config (i.e. works only on the subset of atoms, i.e.  the dopant sites).
                        etmp=[]
                        for i,W in enumerate(all_W): #This is partly slow, only single proc, use coords and all_W as numpy arrays #Use find_equisites from ASE instead? Or make a huge matrix with all_W and do dot products all together.
                            trans=np.zeros((4,4))
                            w=all_w[i]
                            trans[0:3,0:3]=W[:,:]
                            trans[0:3,3]=w[0:3]
                            trans[3,:]=[0.,0.,0.,1.]
                            new_coords=np.dot(trans,coords)
                            #assure the entries btw 0 and 1.  #use the np.where instead. np.where extremely slow compared to below passage.
                            if 0:
                                new_coords=np.where(np.isclose(new_coords,1.),0.0,new_coords)
                                new_coords=np.where(np.isclose(new_coords,0.),0.0,new_coords)
                                new_coords=np.where(new_coords<0. , new_coords+1. , new_coords)
                                new_coords=np.where(new_coords>1. , new_coords-1. , new_coords)
                            else:
                                for j in range(len(conf)):#no of atoms in conf
                                    for k in range(3): #All of the checks are needed for correct symm chek!!
                                        if np.isclose(new_coords[k,j],0.): new_coords[k,j] = 0. 
                                        elif np.isclose(new_coords[k,j],1.): new_coords[k,j] = 0. #to make it comparable to initial ASE-type coords (as 1. is automatically set to 0. in ASE).
                                        elif new_coords[k,j] <0: new_coords[k,j] += 1.
                                        elif new_coords[k,j] >1: new_coords[k,j] -= 1. 


                            if 1: etmp.append(np.unique(new_coords,axis=-1)) # this is a  bit quicker and same results. #axis must be 0
                            else:etmp.append(new_coords) #convert to np.array as well.

                        #End of symm op loop.
                        #"""


                        if 1: all_eSites.append(etmp) #use unique list here ??? original #Check if this is right !!
                        else:
                            flag=0
                            for eS in all_eSites: #Seems like no identical entries in al_eSites.
                                if np.allclose(etmp,eS):
                                    #print ('etmp is in all_eSites',etmp,eS)
                                    flag=1;break

                            if not flag: #etmp stored before
                                all_eSites.append(etmp) 


                    #End of symm. equivalence checking part.

                    if 0: #add ASE compare here
                        SG=ase.spacegroup.Spacegroup(int(sg.split('(')[-1].replace(')','')))
                        etmp=SG.equivalent_sites(X,onduplicates='replace')[0] 

                    #else:
                    degens[cnt]+=1;

                    if args.dopeexcluded:  conf.extend(eList)

                    if args.casm:#create config_list.json
                        occ=[0 for x in range(len(atoms))] #Use real denegeracy values determined.
                        for at in conf: occ[at]=1
                        str11+="""      "%d" : {
                "calctype.default" : {
                  "ref.default" : {
                    "properties" : {
                    }
                  }
                },
                "dof" : {
                  "occupation" : %s
                },
                "selected" : true,
                "source" : [
                  {
                    "enumerated_by" : "ConfigEnumAllOccupations",
                    "step" : %d
                  }
                ]
              },
        """%(cnt,occ,cnt)
                        #outff.writelines(str11)#;outff.flush()


                    #Prepare the full coordinate set for the given config.
                    if args.dType == 'Va' and args.dexcl_type == 'Va':
                        del atoms[[x for x in conf]] #multiple deletion works well.
                    elif args.dType == 'Va':
                        for at in eList: atoms[at].symbol=args.dexcl_type
                        del atoms[[x for x in conf_orig]]

                    elif args.dexcl_type == 'Va':
                        for at in conf_orig: atoms[at].symbol=args.dType
                        del atoms[[x for x in eList]]
                    else:
                        for at in conf: 
                            if at in eList: atoms[at].symbol=args.dexcl_type
                            else: atoms[at].symbol=args.dType

                    #Add the final modifications here!!!
                    if mod_type in ['within','nearest']:
                            ref_list=[atom.index for atom in atoms if atom.symbol==mod_target]  #this is the list of atoms used as reference (i.e. mod_target).
                            sel_list=[] #this is the list atoms that will be modified
                            search_list=[atom for atom in atoms if atom.symbol in mod_repl]
                            if mod_target.lower() in ['do','dopant','dopants']:ref_list=conf
                            cn=0
                            dists0={}
                            for ind1 in ref_list:#indices
                                dists={}
                                #for at in atoms:
                                for  at in search_list:
                                        ind2=at.index
                                        if ind1==ind2:continue

                                        if 1: dist=atoms.get_distance(ind1,ind2,mic=True)
                                        elif 0: dist=atoms[ind1].get_distance(atoms[ind2],mic=True)
                                        else: dist=atoms.get_distance(atoms[ind1],atoms[ind2],mic=True)

                                        key="%s-%s"%(ind1,ind2)
                                        dists0[key]=dist
                                        key=str(ind2)
                                        dists[key]=dist
                                        if mod_type=='within' and dist<=float(mod_value):
                                            if ind2 not in sel_list: sel_list.append(ind2)

                                if mod_type=='nearest':# and dist<=float(mod_value):
                                        if len(dists)==0:continue
                                        keys=list(dists.keys())
                                        keys=sort_dict_by_val(dists)
                                        #sel_list.extend([int(key) for key in keys[0:int(mod_value)]]) #this sometimes yield repeated indices.

                                        for key in keys[0:int(mod_value)]: #fixed.
                                            if int(key) not in sel_list: sel_list.append(int(key))
                                cn+=1

                            if mod_type=='nearest' and len(sel_list) < cn*int(mod_value): #This is needed for adding the closest atom to any of the atoms in the search_list to top up to the requested number of neighbours.
                                keys=list(dists0.keys())
                                keys=sort_dict_by_val(dists0)
                                for ky in keys:
                                    key=int(ky.split('-')[1])
                                    if int(key) not in sel_list: sel_list.append(int(key))
                                    if len(sel_list) == cn*int(mod_value): break

                            if mod_keys[0]=="replace":
                                    if mod_replwith == 'Va':
                                            del atoms[sel_list]
                                    else:
                                            for sel in sel_list:
                                                    atoms[sel].symbol=mod_replwith
                            elif mod_keys[0]=="remove":
                                    del atoms[sel_list]
                    #End of post-modifications

                    #delete the X atoms added for determining the itnerstitial sites in the beginning.
                    if 'X' in args.aType:    del atoms[[atom.index for atom in atoms if atom.symbol=='X']]

                    cnt=str(cnt)
                    if args.sepfol:
                        os.system('mkdir -p %s/%s/'%(outdir,cnt))
                        fname=outdir+"/%s_%d."%(seed,int(cnt))+ext 

                        if args.casm:
                            fname=outdir+'/'+cnt+"/POS"
                            ase.io.write(fname,atoms,format="vasp",direct=True,vasp5=True)
                            #Change the first line reporting atomic types with the config ID.
                            os.system('printf " config_%s \n `tail -n+2 %s` " >%s'%(cnt,fname,fname))
                        elif args.outf: 
                            if args.otype=="vasp": ase.io.write(outdir+'/'+cnt+"/%s"%(args.outf),atoms,format=args.otype,direct=True,vasp5=True)
                            else:ase.io.write(outdir+'/'+cnt+"/%s"%(args.outf),atoms,format=args.otype)
                        else:
                            if args.otype=="vasp": ase.io.write(outdir+'/'+cnt+"/%s_%d."%(seed,int(cnt))+ext,atoms,format=args.otype,direct=True,vasp5=True)#,format=args.otype)
                            else: ase.io.write(outdir+'/'+cnt+"/%s_%d."%(seed,int(cnt))+ext,atoms,format=args.otype)#,format=args.otype)

                        if args.copyinp:#Copy input files from INPUTS dir.
                            if args.casm: 
                                outn='/'.join([outdir,cnt,'calctype.default'])
                                os.system('mkdir -p %s/'%(outn))
                                os.system('cp %s/* %s/'%(inpdir,outn))
                                os.system('cp %s/POS %s/POSCAR'%('/'.join([outdir,cnt]),outn))
                                if args.makepotcar: make_potcar(xc=args.potcarxc,wDir=outn)
                            else:
                                outn='/'.join([outdir,cnt])
                                os.system('cp %s/* %s/'%(inpdir,outn))
                                if args.otype=="vasp" and args.makepotcar: make_potcar(xc=args.potcarxc,wDir=outn)

                    else: #if all out files in the same folder.
                            if args.outf: seed=args.outf
                            #fname=outdir+"/%s_%d."%(seed,int(cnt))+ext #original
                            fname=outdir+"/%s_%d."%(seed.split('_')[0],int(cnt))+ext
                            if args.otype=="vasp": ase.io.write(fname,atoms,format=args.otype,direct=True,vasp5=True)#,format=args.otype)
                            else: ase.io.write(fname,atoms,format=args.otype)

                    cnt=int(cnt)
                    cnt+=1

                    all_configs.append(atoms)
                    all_files.append(fname.split('/')[-1])



            if cnt!=0:
                print("\nTotal of %d symmetrically-unique configurations were generated."%cnt)
            else:
                print ("No configurations were generated. Terminating...");exit()

            if args.ASE_symm: #Our implmenetaion for checking dopant subset of configs is faster than ASE implementaiton of comparing whole structures. This is for doing a last minute check for the pre-filtered set of configs. Sometimes ASE can find symm-equi configs as it also consdiers differetn atomic order and all set of atoms in the comparisons.
                from ase.utils.structure_comparator import SymmetryEquivalenceCheck as ASE_SEC
                print ('compare structures using ASE')
                comp = ASE_SEC()#stol=0.068)
                #print (all_configs)
                ln=len(all_configs)
                uniq_configs=[]#deepcopy(all_configs)
                found=[]
                for i in range(ln):
                    if i in found:continue
                    found.append(i)
                    flag=0
                    for j in range(ln):
                        if i==j or j in found:continue
                        if comp.compare(all_configs[i],all_configs[j]): #1st ref, 2nd compared.
                            print ('%d matches to %d '%(i,j));
                            found.append(j);
                            degens[i]+=degens[j]
                            degens[j]=0
                    uniq_configs.append(i) 
                print('Unique list from ASE: ', uniq_configs)
                all_configs=[all_configs[ii] for ii in uniq_configs] #defgens must be mdified as well. degens of  identical configs must be added.

            if args.casm: 
                str11=str11[0:-2]+'\n'
                str11+="""    }
              }
            }
            """
                outff.writelines(str11)#;outff.flush()
                outff.close();outff2.close()

            ####
            #Now do the actual VASP/CASTEP calculation (SP or geom opt).
            if 1:  
                degens=[d for d in degens if d !=0]
                #print(degens)
                energies=[];sgs=[]

                print ("Total of %d cores will be used in %d parallel tasks."%(args.nprocs,args.ntasks))
                if args.castep: 
                    try:cascmd=os.getenv("CASTEP_COMMAND")
                    except:cascmd=''
                    if cascmd==None:
                        if args.mpirun: cascmd="mpirun -np %s %s"%(args.nprocs/args.ntasks,args.castepexe)
                        else: cascmd='srun -n %d -N %d %s'%(args.nprocs/args.ntasks,max(1,args.nnodes/args.ntasks),args.castepexe)
                    print ("Castep command: ", cascmd)
                elif args.vasp:
                    exe=getenv('VASP_COMMAND')
                    if not exe: 
                        if args.mpirun: exe='mpirun -np %d %s'%(args.nprocs/args.ntasks,args.vaspexe)
                        else: exe='srun -n %d -N %d %s'%(args.nprocs/args.ntasks,max(1,args.nnodes/args.ntasks),args.vaspexe)

                    print("VASP command: ", exe)

                

                def callDFT(inp):
                    i,d,seed=inp
                    #energies=[];sgs=[]
                    #LJ and Morse potentials do not differentiate between the diff atom types. So give the same energies for different configrations. GULP (using force fields for lattices) or LAMMPS or CASTEP/VASP can be used instead.
                    atoms=deepcopy(all_configs[i])
                    fname=all_files[i]
                    del atoms.calc
                    calc=LJ()
                    atoms.set_calculator(calc)
                    e2=atoms.get_potential_energy()
                    del atoms.calc
                    calc=MP()
                    atoms.set_calculator(calc)
                    e1=atoms.get_potential_energy()
                    cwd=getcwd()

                    try:P=atoms.info['pressure']
                    except:P=0.

                    if args.castep: #This can be parallesised over availbale configs
                        del atoms.calc
                        calc = ase.calculators.castep.Castep()
                        print('cwd: ',cwd)

                        if 1:
                            
                            #PP_path=popen('echo "$CASTEP_PP_PATH"',"r").read()[0:-1]

                            calc._castep_command=cascmd

                            #calc._castep_pp_path=PP_path
                            calc._castep_pp_path=os.environ['HOME']+"/APPS/pspots"
                            calc._directory = '%s/CASTEP-tmp'%outdir
                            calc.cell.kpoint_mp_grid = '1 1 1'
                            calc.param.xc_functional = 'PBE'
                            calc.param.cut_off_energy = 300
                            calc.param.max_scf_cycles  = 100
                            calc.param.mix_charge_amp  = 0.1 #for treating slow convergence
                            #calc.param.elec_energy_tol = 1.0e-8
                            #calc.param.perc_extra_bands  = 40
                            #calc.param.finite_basis_corr = 0 #0 by default.
                            calc.param.task = 'SinglePoint'
                            calc.param.mix_charge_amp=0.1
                            calc.param.write_checkpoint ='none'
                            #calc.param.WRITE_CST_ESP=False #option not known to ASE
                            #calc.param.write_bib =False #option not known to ASE
                            #calc.param.bs_write_eigenvalues =0 #band structure file
                            calc._rename_existing_dir = False
                            calc._copy_pspots=True
                            calc._label = 'config_%d'%i
                            atoms.set_calculator(calc)
                            #atoms.calc.set_pspot('00PBE')

                        #try:
                        if inpseed:
                            #read the block species_pot and kpoint_mp_grid info from the ref cell file!!

                            print('Reading CASTEP params from %s/%s'%('..',inpseed))
                            #calc = ase.io.castep.read_seed('%s/%s'%('..',inpseed)) #this does not work
                            #atoms.set_calculator(calc)
                            atoms.calc.merge_param('%s/%s.param'%('..',inpseed))
                            atoms2=ase.io.castep.read_cell('%s/%s.cell'%('..',inpseed))
                                
                            try:
                            #if 1:
                                val=atoms2.calc.cell.species_pot.value
                                val=[ x.split() for x in val.split('\n')]
                                val.pop(0)
                                
                                atoms.calc.cell.species_pot=val
                            except: print('Cannot assign the pspots defined in the reference cell file.')
                            try:atoms.calc.cell.kpoints_mp_grid=atoms2.calc.cell.kpoints_mp_grid.value
                            except: None
                            try:atoms.calc.cell.kpoints_mp_spacing=atoms2.calc.cell.kpoints_mp_spacing.value
                            except: None
                            try:atoms.calc.cell.symmetry_generate=True
                            except:None
                            try:atoms.calc.cell.symmetry_tol=atoms2.calc.cell.symmetry_tol.value
                            except:atoms.calc.cell.symmetry_tol=0.00001
                            atoms.calc.cell.SNAP_TO_SYMMETRY=True

                            #print(atoms.calc.cell.species_pot.value)
                            
                        #except: print('CASTEP params could not be read from %s'%inpdir)
                        #system('cp %s/*.usp .'%cwd)
                        system('mkdir -p %s/CASTEP-tmp/; cp  ../../*.usp %s/CASTEP-tmp/'%(outdir,outdir))
                        #system('cp -v ../../*.usp ')
                        e2=atoms.get_potential_energy()
                        if e2==None: e2=0.0
                        try:P=atoms.info['pressure']
                        except:P=0.

                    elif args.vasp:
                        cwd=getcwd() #StepX

                        print ('\nWorking on ',fname)
                        chdir(outdir)
                        x=fname.split('/')[-1].split('.')[0]
                        seed=x.split('_')[0]+'_'+x.split('_')[-1]
                        system('mkdir -p %s'%seed) 
                        atoms.write('%s/POSCAR'%seed,format='vasp',vasp5=1)

                        #TODO:check if the res file exists, and try to read it.
                        chdir(seed)

                        #print('Working dir: %s'%getcwd())
                        system('cp ../../../%s/INCAR ../../../%s/POTCAR ../../../%s/KPOINTS . &> /dev/null '%(args.idir,args.idir,args.idir)) #replace with cwd
                        #system('cp %s/%s/INCAR %s/%s/POTCAR %s/%s/KPOINTS .'%(cwd,args.idir,cwd,args.idir,cwd,args.idir)) #replace with cwd
                        try: e2,atoms=call_vasp_v2(fname,exe=exe,xc=args.xc,mgms=args.magmoms,hubU=hubU)
                        except Exception as err: 
                            if args.noskip:  print('Problem with DFT calc of %s, (%s) stopping as requested...'%(fname,err));exit()
                            else: print('Problem with DFT calc of %s, (%s)! \nSkipping the config and assign it with 0 energy and P1 symmetry as placeholder...'%(fname,err));chdir(cwd); return (e1,0.0,atoms,'P1 (1)')

                        #ASE does not get the Pressure right when restarting
                        P=float(Popen4("""grep pressure OUTCAR | tail -1 | awk '{print ($4+$9)*0.1}' """)[0][0]) #kB to GPa
                        atoms.info['pressure']=P


                        chdir(cwd)

                    #energies.append(e2) 
                    cell=(atoms.cell, atoms.get_scaled_positions(), atoms.numbers)
                    sg=get_spacegroup(cell, symprec=args.tol)
                    #sgs.append(sg)

                    atoms.info['energy']=e2 #atoms.get_total_energy()
                    atoms.info['name']=seed+'_'+str(cnt)


                    SG=atoms.info.get('spacegroup')

                    #if SG is None or SG == '':   
                    if 1:
                        try:
                            SG=str(get_spacegroup(atoms, symprec=args.tol).symbol.replace(' ',''))
                            atoms.info['spacegroup']=SG.replace('(','').replace(')','')
                        except:
                            SG=str(get_spacegroup(atoms, symprec=args.tol))#.replace(' ',''))
                            atoms.info['spacegroup']=SG.split(' (')[0]

                    stoich=atoms.get_chemical_formula('hill',empirical=1)
                    atoms.info['times_found']=1
                    #atoms.info['name']="%s_%d"%(seed,cnt)
                    atoms.info['name']="%s"%(stoich)
    
                    #print('%s_%d: Ep= %.5f P= %.2f GPa SG= %s'%(seed,cnt,e2,P,atoms.info['spacegroup']))

                    if 0:
                        myres=ase.io.res.Res(atoms)
                        myres.energy=e2
                        myres.write_file('%s.res'%(seed), write_info=0,  significant_figures=6) #works!
                    else:
                        hid=hashid()
                        #fn1='%s-%s-%s.res'%(stoich,seed,hid)
                        fn1='%s-%s-%s.res'%(stoich,'config_enum',hid)
                        fn2='Step%d/%s/%s'%(step,outdir,fname)
                        ase.io.res.write_res(fn1, atoms, write_info=0, write_results=1, significant_figures=6) #works but Energy is always None #ARE THESE the OPTIMISED STRUCTURES CHECK????
                        #ase.io.res.write_res(fn2, atoms, write_info=0, write_results=1, significant_figures=6)
                        #system('cp %s %s'%(fn1,fn2))
                        #ase.io.res.write_res('Step%d/%s/%s_%d.res'%(step,outdir,seed,cnt), atoms, write_info=0, write_results=1, significant_figures=6) #works but Energy is always None


                    #del atoms

                    #return (e2,atoms,energies,sgs)
                    return (e1,e2,atoms,sg)
                    
                    #end of config loop
                    #####

                if len(degens)<args.ntasks: print('Fewer configs to calculate than the set no of parallel tasks (--ntask), reducing no of tasks to 1... '); ntasks=1
                else: ntasks=args.ntasks

                for res in p_imap(callDFT,[ [i,d,seed]  for i,d in enumerate(degens)],num_cpus=ntasks): #[degens[i],all_configs[i],all_files[i]]
                    if res != None:
                        e1,e2,atoms,sg=res
                        energies.append(e2)
                        sgs.append(sg)

                P,Em_bar=boltz_dist(energies,T=args.temp,omega=degens) #probability and configrational free energy. 

                if args.castep or args.vasp:etype='E_DFT'
                else:etype='E_LJ '
                str1+="%s\n"%inpf
                str1+="Config-ID   Degens      %s       E_reduced  Prob.    Space Group\n"%etype+80*'-'+"\n"
                for i,d in enumerate(degens):
                    str1+=" config_%-3d   %-4d  %12.5f  %12.5f  %.3f    %s \n"%(i,d,energies[i],Em_bar[i],P[i],sgs[i])


                if find==len(args.inpf) : print('\n',str1);



                #Add the new configs generated  from the current inpf into seqconfigs (all configs for the current step)
                if args.seq>0: fn=['Step%d/%s/config_%d.res'%(step,outdir,i) for i in range(len(energies))]
                else:fn=['%s/config_%d.res'%(outdir,i) for i in range(len(energies))]

                #try:
                tmp=[fn,Em_bar,[int(sg.split('(')[-1].replace(')','')) for sg in sgs],degens]
                #except: tmp=[fn,Em_bar,[[0 for sg in sgs],degens]]
                for i,t in enumerate(tmp):
                    seqconfigs[i].extend(t)

                #End of input file loop
        print ('End of the file loop for Step %d\n'%step)

        ######
        #Now pick the lucky configs for the next round...

        if args.verb:
            print ("All configs for Step %d"%step)
            for i in range(len(seqconfigs[0])):
              print(seqconfigs[0][i],seqconfigs[1][i],seqconfigs[2][i])

# TO-DO: fix the isue with the configs energies being sorted from larger to smaller (rather than vice versa) when using seqtype == S (symmetric)
        if args.seqtype=='S': #symmetry-based sorting
            #ind=list(np.lexsort((seqconfigs[1],seqconfigs[2]),axis=-1))  #this does not help.
            ind=reversed(list(np.lexsort((seqconfigs[1],seqconfigs[2]),axis=-1))) #both works
            #ind=reversed(list(np.argsort(seqconfigs[2],axis=-1))) #both works
        elif args.seqtype=='E' : #energy-based sorting
            #ind=np.lexsort((seqconfigs[2],seqconfigs[1]),axis=-1)
            ind=np.argsort(seqconfigs[1],axis=-1)
        elif args.seqtype=='D' : #degeneracy-based sorting
            #ind=np.lexsort((seqconfigs[2],seqconfigs[1]),axis=-1)
            ind=reversed(np.argsort(seqconfigs[3],axis=-1))
        elif args.seqtype=='R': #randomise
            ind=list(range(len(seqconfigs)))
            np.random.shuffle(ind)
            ind=[int(i) for i in ind]
            print(ind)

        x=np.array([[seqconfigs[0][i], seqconfigs[1][i], seqconfigs[2][i]] for i in ind])

        str1+="\nAll sorted configs for Step %d\n"%(step)
        str1+='        Config-ID               Energy     SG \n%s\n'%(60*'-')
        for i in x:  str1+='%s %s %s\n'%(i[0],i[1],i[2])
        str1+='\n'

        #Check for duplicate configs generated from different input files at the current step.
        #pAtoms=[]#array of Atoms objects for comparing current config with previous ones.
        pLines=[] #Lines of previous configs
        cnt2=0
        system('touch Picked/Structures-are-not-optimised')
        system('touch ./Structures-are-optimised')
        for i,cf in enumerate(x):#loop over ordered configs.
            if cnt2 ==args.seqpick: break
            #picked.append('../Step%d/%s/config_%d.res'%(step,outdir,int(cf[0])) )#take 2 from user
            flag=0
            if args.seq>0:ln1=open('../'+cf[0]).readlines()
            else: ln1=open(cf[0]).readlines()

            for j,of in enumerate(pLines): #Checking exclty same entries here (can be replaced by symm equivalence checks).
                if ln1==of: str1+='%s matches to %s, skipping...\n'%(cf[0],picked[j]);flag=1;break
                #if filecmp.cmp('../'+cf[0] , '../'+of):print ('%s matches to %s, skipping...'%(cf[0],of));flag=1;break
            if flag==1:continue

            picked.append(cf[0])
            ext=picked[-1].split('.')[-1]
            if args.outf: of=args.outf
            else:of='config'

            atoms=ase.io.read('../%s'%picked[-1]); stoich=atoms.get_chemical_formula('hill',empirical=1)
            #stoich=atoms.get_chemical_formula('hill',empirical=1)

            x=picked[-1].split('/')
            fname="%s-%s"%(x[1],x[2])
            #system('cp ../%s Picked/%s-%s_%d-%s.%s'%(picked[-1],stoich,of,cnt2,hashid(),ext))
            system('cp ../%s Picked/%s'%(picked[-1],fname))
            pLines.append(ln1)
            cnt2+=1

        str1+='\nPicked configs for Step %d: \n%s\n'%(step,'  '.join(picked))

        degen_outf.write(str1+'\n\n'); print (str1)

        if args.seq>0: print ('Done with Step %d...'%step);  os.chdir('../')
        print("Elapsed time for Step %d: %.2f sec.\n"%( step, time.time()-initT))

        degen_outf.close()
        #End of sequential  loop


    print("Total elapsed time: %.2f sec."%( time.time()-initT0))





##############
#DELETED PART#
##############


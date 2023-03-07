#!/usr/bin/env python3

#from matplotlib.pyplot import *
#from numpy import *
import numpy as np
import argparse
from magres.utils import load_all_magres
from magres.atoms import MagresAtoms,MagresAtomsView,MagresAtom
from spglib import *      

import matplotlib
#matplotlib.use('Qt5Agg')
print((matplotlib.get_backend()))
import matplotlib.pyplot as plt

from sys import exit,stdout
from magres.constants import *
import math,re,os
from copy import deepcopy
from scipy.spatial import ConvexHull
from ase import Atoms,Atom,io#,view
from os import popen #,popen4
import ase.io

from ase.visualize import view

from subprocess import Popen,PIPE # as popen # check_output
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


#Definitions from https://easyspin.org/easyspin/documentation/lineshapes.html
def lorentzian(w, w0, gamma):
    return (1/np.pi)*gamma/((w-w0)**2+gamma**2)

def gaussian(w, w0, gamma):
    return ((np.sqrt(np.log(2)/np.pi))/gamma)*np.exp(-np.log(2)*((w-w0)/gamma)**2)

def voigt(alpha):
    sigma = alpha / np.sqrt(2 * np.log(2))

    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

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

def nqr(atom, m): #Computes the Nuclear Quadruploar Resonance, NQR transition freqs in MHz.
    efg = atom.efg
    Q = atom.Q
    spin = atom.spin
    
    Vzz = efg.evals[2]
    vec_zz = efg.evecs[2]
    #eta = (abs(efg.evals[0]) - abs(efg.evals[1]))/efg.evals[2]
    eta =efg.eta #exactly same as above

    A = Vzz * (Q * millibarn) / (4.0 * spin * (2.0*spin - 1.0))
    fq = 3*A * (2.0*abs(m) + 1.0) * math.sqrt(1.0 + eta**2/3)

    return fq

def available_m(spin): #spin quantum no.
     return [m for m in np.arange(-spin, spin+1, 1) if m >= 0.0][:-1]

def get_nqr(atom): #returns the NQR transition frequency averaged over available spin transitions for a given MagresAtom.
    freqs = []
    for m in available_m(atom.spin):
        freq_MHz = nqr(atom, m) / megahertz
        freqs.append(freq_MHz)  #original
        #freqs.append(abs(freq_MHz))

    return np.mean(freqs)

def quad_shift(atom,larm=100): #compute the total shift, including isotropic,second and fourth rank anisotropic quadrupolar shifts (?). P_Q is also called second-order quadrupolar effect.
    #larm:Larmor frequency (def: 100 MHz).

    efg = atom.efg
    Q = atom.Q
    spin = atom.spin
    m=spin%1
    Vzz = efg.evals[2]
    vec_zz = efg.evecs[2]
    #sigma_iso=atom.sigma
    #eta = (abs(efg.evals[0]) - abs(efg.evals[1]))/(efg.evals[2]-sigma_iso)
    eta =efg.eta #exactly same as above
    Cq=efg.Cq
    Pq = (Cq*1.0e-6)*math.sqrt(1.0+(eta*eta)/3.0);
    I=spin
    try:qiso_shift=(-(3.0/40.0)*(math.pow((Pq/larm),2.0))*(I*(I+1)-9.0*m*(m-1)-3.0)/(I*I*math.pow((2*I-1),2.0))*1.0e6)*1e12 #last one added for unit compatibility.
    except:qiso_shift=-999.

    return qiso_shift

def quad_shift2(atom,larm=100): #compute the total shift, including isotropic,second and fourth rank anisotropic quadrupolar shifts.
    #larm:Larmor frequency (def: 100 MHz).

    #DOES NOT WORK CORRECTLY!! USE the other one.

    efg = atom.efg
    Q = atom.Q
    spin = atom.spin
    m=spin%1
    
    Vzz = efg.evals[2]
    vec_zz = efg.evecs[2]
    #eta = (abs(efg.evals[0]) - abs(efg.evals[1]))/efg.evals[2]
    eta =efg.eta #exactly same as above

    A = Vzz * (Q * millibarn) / (4.0 * spin * (2.0*spin - 1.0))
    fq = 3*A * (2.0*abs(m) + 1.0) * math.sqrt(1.0 + eta**2/3)
    
    qiso_shift=(fq**2/larm)*A*(1.0 + eta**2/3)
    print(qiso_shift)
    return qiso_shift

#def coord_anlys(atom,max_dist=3.0,cov_radii={}):
def coord_anlys(atom,atoms,neig_list=[],max_dist=None,cov_radii={},tol=15.,BVS_ref={}):
    #Input is a MagresAtom,max_dist: cutoff distance to determine neighbours; radii: table of covalent radii of atoms;negi_list
    #print (atom)#,[at.species ])
    #for at in atoms:        print at,at.index

    #print atom,max_dist
    #try:
    atoms_coll=[Atom(atom.species,atom.position)] #add central atom.
    ids=[atom.index]
    #except: None
    #atoms_coll=[atom]

    #atoms_coll=MagresAtom(atom)

    dists=[];angles=[]
    if len(neig_list)==0:
        neig_list=[x for x in atoms.within(atom, max_dist) if x != atom] #atoms is read as global variable !!
    #neig_list=np.array(neig_list)
    neig_list_cp=deepcopy(neig_list)
    coord=len(neig_list)
    vert_dists=[]
    bvs=[]
    del_list=[]
    max_dist_orig=max_dist
    #scpos=atoms.get_scaled_positions()
    #for atom_X in neig_list:
    for i in range(coord):  #iterate over coordination no.
        atom_X = neig_list_cp[i]
        #if atom == atom_X: continue #no need as done before.
        if max_dist_orig==None: max_dist=(cov_radii[atom.label]+cov_radii[atom_X.label])*1.2 #10% margin.
        dist=atoms.dist(atom, atom_X)  #Does this include the PBC ???Check !!
        if len(neig_list)>0 and dist>max_dist:print(dist);del_list.append(i); continue #neig_list.pop(i); continue #Filter out the distances larger than typical bond length for the given atom pair.
        dists.append(dist)
        try:            atoms_coll.append(Atom(atom_X.species,atom_X.position))#,scpos[atom_X.index]))
        except: atoms_coll.append(atom_X)#atoms_coll+=atom_X#
        ids.append(atom_X.index)

        #Get BVS value
        try: 
            if atom.species=="O":bref=BVS_ref[atom_X.species][atom.species] #bvs.append(np.exp((BVS_ref[atom_X.species][atom.species]-dist)/args.bvs_B))
            else:bref=BVS_ref[atom.species][atom_X.species] #bvs.append(np.exp((BVS_ref[atom.species][atom_X.species]-dist)/args.bvs_B)) 

            #print "BVS_ref:",bref
            bvs.append(np.exp((bref-dist)/args.bvs_B))
        except:bvs.append(0.0)

        #for atom_Y in neig_list: #Loop for angles
        for j in range(i+1,coord):
            atom_Y=neig_list_cp[j]
            if atom_Y != atom_X:
                #max_dist=(cov_radii[atom_Y.label]+cov_radii[atom_X.label])*1.2 #10% margin.
                vdist=atoms.dist(atom_Y,atom_X)
                #if vdist>max_dist: print "burda",vdist,max_dist;continue #Do not take the vertices that are directly bonded. THIS DOES NOT WORK FOR POLYHEDRA VERTICES !!
                try:angle=atoms.angle(atom_X, atom, atom_Y, degrees=True)
                except:angle=np.nan
                if not np.isnan(angle) and 180.-tol > angle < 180.+tol:
                    angles.append(angle) #only non-NaN angles. Do not take the opposite vertices.
                    vert_dists.append(vdist)
    #In principal there shouldn't be any need for deleting neighbours, as they determined in the beginning using the general max_dist argument.
    #for ii in range(len(del_list)): 
    #    print neig_list[ii],del_list[ii]
    #np.delete(neig_list,del_list)

    coord=len(neig_list)#update in case neig_list modified.
    dm=np.mean(dists);am=np.mean(angles); vm=np.mean(vert_dists)
    dstd=np.std(dists);astd=np.std(angles); vstd=np.std(vert_dists)

    #Compute Bond Length Distortion (BLD, delta)
    summ=0.
    for i in dists:
        summ+=((i-dm)/dm)**2
    BLD=summ/float(coord)
    #BLD=(sum([(i-dm)/dm  for i in dists])**2)/coord

    #Bond-angle distortion/variance (BAD, sigma^2)
    try:
        if coord==4: x=5;a0=109.4712 #tetrahedron case
        elif coord==6: x=11;a0=90.0 #octahedron case.
        summ=0.
        for ai in angles:
            summ+=((ai-a0))**2
        BAD=summ/float(len(angles)-1)
    except: BAD=None
    #BAD=(sum([(ai-a0)/am  for ai in angles])**2)/(len(angles)-1)
    
    #Polyhedral volume
    points=[]
    #for n in neig_list: points.append(n.position)
    points=np.array([n.position for n in neig_list])
    try:V=ConvexHull(points).volume
    except:V=0.0
    try:S=ConvexHull(points).area#surface area
    except:S=0.

    #Distrotion index (DI) of bond lengths 
    DIB=(sum([abs(di-dm)  for di in dists]))/coord/dm # as in Acta Cryst. (1974). B30, 1195 (VESTA implementaion also uses this)
    #DIB=(sum([(di-dm)**2/dm**2  for di in dists]))/coord # as in  Ferrara et al. JPCC 2013

    #Distrotion index (DI) of angles 
    DIA=(sum([abs(ai-am)  for ai in angles]))/len(angles)/am # as in Acta Cryst. (1974). B30, 1195 
    #DIA=(sum([(ai-am)**2/am**2  for ai in angles]))/len(angles)  # as in  Ferrara et al. JPCC 2013

    #Distrotion index (DI) of edge lengths 
    DIE=(sum([abs(vi-vm)  for vi in vert_dists]))/len(vert_dists)/vm # as in Acta Cryst. (1974). B30, 1195 
    #DIE=(sum([(vi-vm)**2/vm**2  for vi in vert_dists]))/len(vert_dists) # as in  Ferrara et al. JPCC 2013

    #Quadratic elongation (check)
    try:
    #ideal=cov_radii[atom.species]+cov_radii[atom_Y.species] #not correct use volume instead.
    #print ideal
        summ=0.
        if coord==4: A=math.sqrt(2)/12.
        elif coord==6:A=1
        elif coord==8: A=math.sqrt(2)/3.
        elif coord==12: A=(15+7*math.sqrt(5))/4.
        elif coord==20: A=5*(3+math.sqrt(5))/12.
    
        #V=A*(vm**3) #avg. vertice dist.
        ideal=math.pow(V/A,1./3.) #As this formula is defined for edge lengths (vertice dists), does not work for center-vertice distances (i.e. dists) as opposed how is defined in VESTA.
        #for di in dists: summ+=(di/dm)**2 #Must use ideal bond distance computed from the actual polyhedorn volume instead of avg. center-vertice distance.
        #QE=summ/float(coord)
        for vi in vert_dists: summ+=(vi/ideal)**2
        QE=summ/float(len(vert_dists))
    except: QE=None 

    #Effective coordination number
    ECoN=0.
    sum1=0.;sum2=0.
    for di in dists:
        sum1+=di*math.exp(1-(di/min(dists))**6)
        sum2+=math.exp(1-(di/min(dists))**6)
    lav=sum1/sum2
    for di in dists: ECoN+=math.exp(1-(di/lav)**6)

    #Print out the analysis results.
    #print "\nPolyhedron distortion analysis of %s with %d-fold coordination within %.1f A."%(atom,coord,max_dist)
    print("\nPolyhedron distortion analysis of %s with %d-fold coordination"%(atom,coord))
    print("Bonds and Bond Valance:")
    for i in range(coord):print("%s-%s: %.5f A / %.5f"%(atom.species+str(atom.index),neig_list[i].label+str(neig_list[i].index),dists[i],bvs[i]))
    print("Mean bond length = %.3f+-%.3f A"%(dm,dstd))
    print("Bond Valance Sum (BVS) = %.3f "%(sum(bvs)))
    print("Angles: %s"%(", ".join(["%.4f"%i for i in angles])))
    #for i in angles: print i,
    #print
    print("Mean angle = %.1f+-%.1f deg"%(am,astd))
    print("\nPolyhedron volume: %.4f A^3"%V)
    print("Bond-length distortion (delta): %.4f "%BLD)
    if BAD!=None: BADstr="%7.4f deg^2"%BAD
    else:BADstr="N/A"
    print("Bond-angle distortion/variance (sigma^2): %13s"%BADstr)
    #else:  print "Bond-angle distortion/variance (sigma^2): Not defined"
    print("Distortion index (DI) of bond lengths: %.5f "%DIB)
    print("Distortion index (DI) of angles: %.5f "%DIA)
    print("Distortion index (DI) of edge lengths: %.5f "%DIE)
    if QE!=None: QEstr="%-10.4f"%QE
    else: QEstr="N/A"
    print("Quadratic elongation (lambda) of edges: %10s"%QEstr)
    #else: print "Quadratic elongation (lambda) of edges: Not defined"
    print("Effective coordination number: %.4f"%ECoN)
    print()

    #str1="%-6s %2d %.3f +- %-.3f  %5.1f +- %-5.1f  %7.4f  %.4f  %.4f  %-10s %.5f %.5f %.5f   %-10s %-.4f\n"%(atom,coord, dm,dstd,am,astd,sum(bvs),V,BLD,BADstr.split()[0],DIB,DIA,DIE,QEstr,ECoN)
    str1="%.3f +- %-.3f  %5.1f +- %-5.1f  %7.4f  %.4f  %.4f  %-10s %.5f %.5f %.5f   %-10s %-.4f\n"%(dm,dstd,am,astd,sum(bvs),V,BLD,BADstr.split()[0],DIB,DIA,DIE,QEstr,ECoN)
    #outf_ca.write(str1)

    #print atoms_coll
    try:        return atoms_coll,str1,ids #list of Atoms objects and their orig AtomIds for the main atom and the neighbours.
    except:        return atoms_coll,str1

def boltz_dist(energies,T=298.15,omega=[]):#Return the occupation probabilities of the configurations at a given temperature based on their energies.
    kb= 8.6173303*10**-5 #Boltzmann constant (eV/K).
    if len(omega)==0:#If the degeneracies are not given explciitly, all set to 1.
        omega=[1 for E in energies]

    if 1: #Get relative energies.  Doesn't really matter as long as you use the normalised factors.
        mn=min(energies)
        energies=[i-mn for i in energies]
    probs=[]
    for E in energies:
        probs.append(math.exp(-E/kb/T))
    #Normalise    
    Z=sum(probs) #i.e. partition fnc
    probs=[Pn/Z for Pn in probs]

    #Configurational statistics as given in R. Grau-Crespo et al. J.Phys: COndens. Matter, 19,2007,256201
    print("\nTemperature: %d K"%T)
    E_avg=sum([energies[i]*probs[i] for i in range(len(energies))])
    print("Average energy of the sytem in configurational equilibirum,  E=%.5f eV"%E_avg)

    F=-kb*T*np.log(Z)
    print("Configurational free energy in the complete space, F=%.5f eV"%F)

    S= (E_avg-F)/T
    print("Configurational entropy in the complete space, S=%.5f eV/K"%S)

    Smax=kb*np.log(len(energies))
    print("Upper limit of config. entropy, Smax= %.5f eV/K"%Smax)

    #Now count in the degenaricies of the configs.
    Sm=[kb*T*np.log(om) for om in omega] #degeneracy entropy

    #for i,E in enumerate(energies):
    Em_bar=[energies[i]-T*Sm[i] for i in range(len(energies))]

    Pm=[np.exp(-Em_bar[i]/kb/T) for i in range(len(energies))] 
    Z_bar=sum(Pm)
    #Pm_bar=[(1/Z)*np.exp(-Em_bar[i]/kb/T) for i in range(len(energies))] #reduced  probability for an independent config.
    Pm_bar=[P/Z_bar for P in Pm]

    E_avg=sum([Em_bar[i]*Pm_bar[i] for i in range(len(energies))])

    F=-kb*T*np.log(Z_bar)
    print("Configurational free energy in the reduced config. space, F=%.5f eV"%F)

    S= (E_avg-F)/T
    print("Configurational entropy in the reduced config. space, S=%.5f eV/K"%S)

    #print "Reduced probabilties for  independent configurations: ",Pm_bar

    return Pm_bar#,E_avg

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--inpf", type=str, default=["input.magres"],nargs="*",                    help="Input file to read the MAGRES data from.")

    parser.add_argument("-larm","--larm", type=float,default=700.13,
                    help="Larmour frequency (MHz) to compute the  quadrupolar effects. Settign the value for 1H (Def: 700.13 MHz)")

    parser.add_argument("-md","--max_dist", type=float,#default=3.0,
                    help="Max. distance to search for neighbours.")

    parser.add_argument("-rs","--ref_shift", type=float,default=0.0,
                    help="Reference shift for computing chemical shifts.Def: no shift")

    parser.add_argument("-at","--atype", type=str,required=True,#default="O",
                    help="Atom type for which NMR peaks will be computed and coordination analysis will be run.")

    parser.add_argument("-qs","--quad_shift", action='store_true', default=False, help="To shift isotropic shieldings by isotropic quadrupolar shifts. Def: False (i.e. iso. shieldings (delta_iso) is reported.")  #This is not correct, delete this !!!

    parser.add_argument("-quad","--quad", action='store_true', default=False, help="To include the 1st and 2nd-order quadrupolar effects in the NMR spectrum. Def: False (i.e. only convoluted iso. shieldings (delta_iso) are plotted.")

    parser.add_argument("-np","--no_plot", action='store_true', default=False, help="To supress plotting of NMR peaks.  Def: spectra are plotted  ")

    parser.add_argument("-sl","--show_lines", action='store_true', default=False, help="To plot of NMR peak lines (isotropic shieldings/chemical shifts).  Def: Peaks and their Gaussian&Lorentzian broadenings will be plotted. ")

    parser.add_argument("-cl","--color_lines", action='store_true', default=False, help="To color NMR peak lines with respect to coordination no of the atom.  Def: Only black lines. ")

    parser.add_argument("-nl","--no_legend", action='store_true', default=False, help="Do not show the plot legend.")

    parser.add_argument("-ov","--overlay", action='store_true', default=False, help="To treat the input files seperately and overlay their NMR spectra. Def: Corresponding NMR spectra are convoluted together. ")

    parser.add_argument("-mono","--mono", action='store_true', default=False, help="Monochromic ovelay plots (i.e. individiual contributions). ")

    parser.add_argument("-save","--save", action='store_true', default=False, help="Plotted data will be saved to separate dat files")

    parser.add_argument("-boltz","--boltz", type=float, nargs="*",default=[0.], help="To weigh the NMR peaks from multiple configurations using Boltzmann distribution at a given temperature [K] based on their formation energies to get a convoluted NMR spectrum. Def: no weighting is done. Usage: -boltz TEMP")
    parser.add_argument("-pa","--EperAtom", action='store_true', default=False, help="Consider the relative energies per atom rather than total (relative) energies. Def: Relative total energies used.")
    #parser.add_argument("-fu","--Eperfu", action='store_true', default=False, help="Consider the relative energies per fu rather than total (relative) energies. Def: Relative total energies used.")
    parser.add_argument("-fu","--Eperfu", type=int, const=1,default=None,nargs="?",help="Consider the relative energies per fu rather than total (relative) energies. fu value can be set explicitly by using -fu X; otherwise it's auotmatically determined. Def: Relative total energies used.")

    parser.add_argument("-bt","--bType", type=str, choices=['Gau','Lor','both'],default="both", help="Convolution/broadening type. Def: both")

    parser.add_argument("-gamma","--gamma", type=float, default=2.0, help="Lorentzian/Gaussian broadening factor (Gamma). Def: 2.0 ppm")

    parser.add_argument("-minmax","--minmax", type=float, nargs=2, default=0., help="The xrange of the convoluted NMR spectrum. Def: Auto range. Usage: -minmax X Y.")

    parser.add_argument("-ref","--ref_data", type=str,nargs="*", help="To load experimental reference data to overlay the computed NMR spectrum. Input should be in ASCII format (peak position, intensity). Multiple ref files supported.")
    
    parser.add_argument("-skref","--skip_ref", type=int, default=1, help="Skip every X data points from the reference spectrum. Def: no data skipped.")

    #parser.add_argument("-shref","--shift_ref", type=float, default=0., help="Shift the reference spectra along y-axis by X%. for a better visibility. Def: No shift.")

    parser.add_argument("-sep","--separate", action='store_true', default=False, help="Seperate the reference spectra from the computed ones using subplots. Def: Computed and reference data are overlaid.")

    parser.add_argument("-ca","--coord_anlys", action='store_true', default=False, help="Perform detailed cordination distortion analysis for the chosen atom type.")

    parser.add_argument("-cna","--cna", action='store_true', default=False, help="Perform detailed closest neighbour analysis for the chosen atom type within the --max_dist Angstroem.")


    parser.add_argument("-bvs_B","--bvs_B", type=float, default=0.37, help="Emprical constant, B value for computing the Bond Valance Sum (BVS) for the selected atom types. Def: 0.37 A ")

    parser.add_argument("-bvs_R0","--bvs_R0", type=float, help="Bond valance parameter, R0 value for computing BVS for the selected atom types. No default: if no value given then --bvs_file will be used to obtain R0 values. ")

    parser.add_argument("-bvs_file","--bvs_file", type=str, default=os.environ['HOME']+"/bvparm2013.cif", help="File containing R0 values for computing BVS for the selected atom types (formatting taken from VESTA code pack). If  multiple enetries for given caton/anion pair, then the first entry will be used. Def: ~/bvparm2013.cif")

    parser.add_argument("-bvs_ox","--bvs_ox", type=int, help="Oxidation state for the central atom (cation). If no input given the first available state is used by default. ")

    parser.add_argument("-sp","--save_pol", action='store_true', default=False, help="Save all polyhedra from the collection of input files into a single file (out.res) for visualisation purposes. Def: False")
    
    parser.add_argument('-tol', '--tol',type=float, default=1e-3,help="The symmetry tolerance. Def: 1e-3")

    parser.add_argument("-S","--silent", action='store_true', default=False, help="To surpress the NMR peak analysis output on the stdout, good for plotting convoluted NMR spectra from many magres files. Def: Detailed analysis is shown.")

    parser.add_argument("-v","--verb", action='store_true', default=False, help="Verbose output")
    

    args = parser.parse_args()

    args.inpf=sorted_nicely(args.inpf)


    if args.separate and not args.ref_data: print("-sep option requires -ref input. Terminating...");exit()

    #Do not forget to change this for other systems.
    atList=['O','La','Zr','Li','Al','Ga','S','P'] #TO-DO:mujst find the all atom types present in the system, or takeas user input.
    #atList=['O']
    target=args.atype

    larm=args.larm #Larmor frequency.
    if args.max_dist:
        max_dist=args.max_dist
    else:#max dist will be determined individually based on tabulated atomic covalent radii later on.
        max_dist=0
            
    ref_shift=args.ref_shift
    #atoms.species('O').set_reference(30) #Def: 0
    #print atoms.species('O').reference

    cov_radii={}
    for ln in open(os.environ['HOME']+"/covalent_radii.dat",'r'):
        x=ln.split()
        cov_radii[x[0]]=float(x[1])


    #Read the relevant R0 data for the atype from the BVS reference file.
    BVS_ref={} #[prim_atom][secondary_atom]
    if args.bvs_R0:bvs_R0= args.bvs_R0
    else: 
        with open(args.bvs_file) as inpf:
            for ln in inpf:
                #format:     
 #Ac 3   O  -2    R0    B     Ref   Notes
                if re.search( r'[A-Z][a-z]*',ln[0:2]) :#args.atype[0]+atList: 
                    #print ln
                    x=ln.split()
                    #if len(x)!=8:continue
                    if x[0] not in args.atype and x[0] not in atList: continue #not only the primary atom type, but also potential neighbor types included (needed for O or other anions).
                    if float(x[5])!=args.bvs_B:continue #only take R0 values for correct B parameter.
                    if args.bvs_ox and int(x[1])!= args.bvs_ox: continue #only take the desired ox. state if given by user.
                    #finally collect the relevant data.
                    if x[0] not in BVS_ref: BVS_ref[x[0]]={}
                    if x[2] not in BVS_ref[x[0]]: BVS_ref[x[0]][x[2]]=float(x[4]) #only take first entry.
    if args.verb: print("BVS ref dict: ", BVS_ref)

    if args.save_pol:
        atoms_coll=[Atoms('H', pbc=True,positions=[(0,0,0)])]#initialise atoms object.
        del atoms_coll[0][0]
        args.coord_anlys=1

    if args.cna:outf_cna=open("cna.dat",'w');outf_nNMR=open("neighNMR.dat",'w');
    if args.coord_anlys: 
        outf_ca=open("ca.dat",'w');
        outf_ca.write("#AtomId CoordNo, Mean bond length, Mean bond angles, BVS, PolyVol, BLD, BAD, DIB, DIA, DIE, QE, Eff.CoordNo \n")

    all_shifts=[];energies=[];pstoich="";SGs=[]
    #for inpf in args.inpf:
    for ind,inpf in enumerate(args.inpf):
        print("\nWorking on %s..."%inpf)
        if args.cna:outf_cna.write("\n# %s\n"%inpf)
        if args.coord_anlys: outf_ca.write("# %s\n"%(inpf))
        try:
            atoms = MagresAtoms.load_magres(inpf)
            atoms_orig=ase.io.read(inpf)
        except:
            continue

        scaled_positions= atoms_orig.get_scaled_positions()
        cell=(atoms_orig.cell, scaled_positions, atoms_orig.numbers)
        sg=get_spacegroup(cell, symprec=args.tol)
        print(("Space group= %s with tol=%.2e"%(sg,args.tol)))
        SGs.append(sg)
        #lattice, scaled_positions, numbers = find_primitive(cell, symprec=args.tol)

        #x=inpf.split("/")[-1].split(".")[0]+".castep"
        x=inpf.replace(".magres",".castep")
        try:
            skey="Final free energy (E-TS)"  #gets the H not E)
            #e=float(popen4('grep  "%s" %s | tail -n 1'%(skey,x),'r')[1].readlines()[0].split()[-2]) #If SCF converged then should be there.
            e=float(popen('grep  "%s" %s | tail -n 1'%(skey,x),'r').read()[0:-1].split()[-2]) #If SCF converged then should be there.
        except:
            skey="Total energy corrected for finite basis set"  #If IPRINT=2 in param.
            #e=float(popen4('grep  "%s" %s | tail -n 1'%(skey,x),'r')[1].readlines()[0].split()[-2])
            e=float(popen('grep  "%s" %s | tail -n 1'%(skey,x),'r').read()[0:-1].split()[-2])

        reduced,fu=getReduced(ase.io.read(inpf))
        print("%s with fu=%d"%(reduced,fu))


        #stoich=atoms_orig.get_chemical_formula(mode='hill')
        stoich=reduced
        if ind>0 and stoich!=pstoich: print("Warning: combining spectra calculated using systems with diverse contents.", stoich, pstoich)
        pstoich=stoich

        if args.Eperfu :
            if args.Eperfu>1:fu=int(args.Eperfu); print("fu set to %d by user."%fu)
            #else:fu=fu; print "fu set to: %d"%fu

        #Boltzmann weights really depend on the whether E per atom or total E per fu  is used.
        if args.EperAtom:       energies.append(e/len(atoms))#per atom
        elif args.Eperfu: energies.append(e/fu)
        else: energies.append(e)


        #print atoms.lattice
        if len(atoms.species(target))==0 :
            print("Target atoms (%s) is not present in the system. Skipping..."%target);continue#exit()
        crds={} #list of coordination nos.
        conn={};conn2={};conn3={} #connectivity data
        Cqs={} #Quadrupolar couplings
        shifts={};etas={};nqrs={};qshifts={}
        dists = []; 

        #Initialise atoms object
        if args.save_pol:
            atom_coll=Atoms('H', pbc=True,positions=[(0,0,0)])#initialise atoms object.
            del atom_coll[0]

        for atom in atoms.species(target):
            cnt=0; all_dists=[];angles=[]
            str1=""; str2=""
            min_dists=[];            min_atoms=[]
            for at in atList: #list of potential neighbour types
                try:
                    dists=[]
                    if not args.max_dist: #max_dist==0: 
                        max_dist=(cov_radii[atom.label]+cov_radii[at])*1.2 #10% margin.
                        if args.verb: print("Max distance for determining neighbours (%s-%s) is automatically set to : %.2f A"%(atom.label,at,max_dist))

                    neig_list=atoms.species(at).within(atom, max_dist)
                    if len(neig_list)==0 :
                        #if  args.verb: print 'No closest neighbour found within %.2f A for %s-%s pairs'%(max_dist,atom,at); 
                        continue

                    coord=len(neig_list)                    

                    for i in range(len(neig_list)):  
                        atom_X = neig_list[i]
                        if atom == atom_X: continue #no need as done before.
                        cnt += 1
                        dist=atoms.dist(atom, atom_X)
                        min_dists.append(dist); min_atoms.append(atom_X)

                        """#Ascending order wrt distances
                        if len(min_dists)>0: 
                            for k in range(len(min_dists)):
                                if  dist <= min_dists[k]:# and atom_X not in min_atoms:
                                    min_dists.insert(k,dist)
                                    min_atoms.insert(k,atom_X)
                                    #break
                                elif atom_X not in min_atoms:
                                #else:
                                    min_dists.append(dist); min_atoms.append(atom_X)
                        elif atom_X not in min_atoms:
                            min_dists.append(dist); min_atoms.append(atom_X)
                        """

                        dists.append(dist)
                        all_dists.append(dist)

                        #loop for angles
                        for j in range(i+1,coord):
                            atom_Y=neig_list[j]
                            if atom_Y != atom_X:
                                angle=atoms.angle(atom_X, atom, atom_Y, degrees=True)
                                if not np.isnan(angle): angles.append(angle) #only non-NaN angles.

                    if len(dists)!=0:
                        str1+="%dx%s, %4.2f +- %4.2f; "%( len(dists),at,np.mean(dists),  np.std(dists))

                except:  continue

            if len(angles)>0: str1+="A: %4.2f +- %4.2f"%( np.mean(angles),  np.std(angles))
            else:str1+="A: "
            if args.cna: #args.verb:
                if args.verb: str1+="\nClosest neighbour (Id, Distance, Delta_iso, Cq and eta_q \n"
                str2+='CN: ' 
                for j,ma in enumerate(min_atoms): 
                    strx1="%6s %6.4f %8.2f %6.2f %5.2f %.5f\n"%(ma.species+str(ma.index), min_dists[j],(args.ref_shift- ma.ms.iso),abs(ma.efg.Cq),ma.efg.eta,energies[-1])
                    #strx1="%6s %6.4f %8.2f %6.2f %5.2f\n"%(ma.species+str(ma.index), min_dists[j],(args.ref_shift- ma.ms.iso),abs(ma.efg.Cq),ma.efg.eta)
                    if args.verb: str1+=strx1
                    outf_nNMR.write(strx1)

                    str2+="%6s %.2f"%(ma.species+str(ma.index), min_dists[j])
                    
                str2+="\n"
            #outf_cna.write(str2)

            #ms:magnetic-shiledings, efg:electric field gradient.
            atomID=atom.label+str(atom.index)
            crds["%s"%atomID]=cnt
            Cqs["%s"%atomID]=atom.efg.Cq
            etas["%s"%atomID]=atom.efg.eta
            nqrs["%s"%atomID]=get_nqr(atom)
            qshifts["%s"%atomID]=quad_shift(atom,larm=larm)
            conn["%s"%atomID]=str1
            conn2["%s"%atomID]=str2
            str2=''; str3=''
            if args.quad_shift:
                shifts["%s"%atomID]=atom.ms.iso-qshifts["%s"%atomID] #MagresView seems to use isotrpic quadrupolar shifts (subtracting from iso shieldings)
            else:
                 shifts["%s"%atomID]=atom.ms.iso

            if args.ref_shift: shifts["%s"%atomID]=ref_shift-shifts["%s"%atomID]

            if args.coord_anlys: 
                if args.max_dist:max_dist=args.max_dist
                else:
                    try:max_dist= max(all_dists)*1.2 #should be 1.1 !!
                    except:max_dist=None
                #print "Max distance for determining neighbours: %.2f A"%max_dist
                res,str3,ids=coord_anlys(atom,atoms,max_dist=max_dist,cov_radii=cov_radii,BVS_ref=BVS_ref)
                conn3["%s"%atomID]=str3
                if args.save_pol:
                    neig_dict={} #O-neighbour dict for the neighbours of the central atoms.
                    res=Atoms(res,pbc=1)
                    magres_ats=[]
 
                    for k,r in enumerate(res):
                        neig_dict["%s%s"%(r.symbol,r.index)]=[]
                        if r.symbol=="O":      continue
                        cid=ids[k]
                        #print r,atoms[r.index],atoms[cid],atoms.get(r.symbol,cid)      #we need to use the real index, assigned in the coord_anlys function !!!! 
                        max_dist=args.max_dist
                        if not args.max_dist: max_dist=(cov_radii[r.symbol]+cov_radii['O'])*1.2 #10% margin.

                        gec=[x for x in atoms.within(atoms.get(r.symbol,cid), max_dist) if ( x != atoms.get(r.symbol,cid) and x.species=="O" ) ] 
                        labels=["%s%s"%(at.label,at.index) for at in magres_ats]
                        for g in gec:
                            #print g
                            if "%s%s"%(g.label,g.index) not in labels: 
                                magres_ats.append(g)
                                g=Atom(g.species,g.position)
                                neig_dict["%s%s"%(r.symbol,r.index)].append(g)

                    atom_coll.set_cell(atoms.lattice);                    res.set_cell(atoms.lattice)
                    scpos=res.get_scaled_positions(wrap=False)
                    for k,r in enumerate(res):
                        sp=scpos[k]
                        sp0=scpos[0]
                        neighs=Atoms(neig_dict["%s%s"%(r.symbol,r.index)],pbc=1)
                        neighs.set_cell(atoms.lattice)
                        scpos2=neighs.get_scaled_positions(wrap=0)

                        for j in range(3):
                            diff=sp[j]-sp0[j]
                            if np.fabs(diff)>0.5: 
                                scpos[k][j]=(scpos[k][j] - np.sign(diff))
                                for l in range(len(neighs)): #udapte the O-neighbor of the given neighbour of the central atom.
                                    scpos2[l][j] =scpos2[l][j] - np.sign(diff)
                        neighs.set_scaled_positions(scpos2)
                        neig_dict["%s%s"%(r.symbol,r.index)]=neighs

                    res.set_scaled_positions(scpos)
                    #print res,neighs
                    atom_coll.extend(res)
                    for j in list(neig_dict.values()): atom_coll+=j
                    #view(atom_coll)

                    #Check and delete duplicate atoms
                    toDel=[]
                    for i,at in enumerate(atom_coll):
                        if i==0:continue
                        pos1=at.position
                        for j in range(i-1,0,-1):
                            pos2=atom_coll[j].position
                            if np.isclose(pos1[0],pos2[0]) and np.isclose(pos1[1],pos2[1]) and np.isclose(pos1[2],pos2[2]): toDel.append(i); continue #use np.dot((A*B).T,A*B)
                    del atom_coll[toDel]
                    atom_coll.write("%s-polyhedra.xyz"%(inpf.split(".")[0]),format="xyz")
                    atom_coll.write("%s-polyhedra.cif"%(inpf.split(".")[0]),format="cif")
                    atoms_coll.append(atom_coll)


        #print "\nMean bond length =", np.mean(dists), "+-", np.std(dists)

        #One can use atoms.atomID as well.

        all_shifts.append(shifts)
        keys=list(shifts.keys())
        if 1:#sort shifts ascending.
            keys=sort_dict_by_val(shifts)
        else:
            #keys.sort()
            keys=sorted_nicely(keys)

        for atom in keys:
            if not args.silent:
                #print "atom,coord,Shielding,Quad Shift, Quad Coupling Cnst(Cq), EFG asymm (eta_Q), Avg. NQR freq., Neigbours (within %.1f A) with average bond lengths and angles."%max_dist
                str3= "%4s %2d %8.2f %6.2f %5.2f %5.2f %6.3f; "%(atom,crds[atom],shifts[atom],qshifts[atom],abs(Cqs[atom]),etas[atom],nqrs[atom])
                print(str3+"B: %s"%(conn[atom]))

            if args.cna:outf_cna.write(str3+conn2[atom]) #TODO: can be filtered based on Boltz Weigth !!
            if args.coord_anlys: outf_ca.write(str3+conn3[atom])

    if args.save_pol: 
        coll=Atoms("H")
        del coll[0]
        for j in atoms_coll:            coll+=j #combine all polyhedra from each .magres file.
        coll.write('all_polyhedra.xyz',format='xyz')
        coll.write('all_polyhedra.cif',format='cif')
        #atom_coll.write('out.vasp',format='vasp',vasp5=1)

    if args.cna: outf_cna.close()
    if args.coord_anlys:outf_ca.close()

    if args.boltz != [0] and args.boltz != 0: #If Boltzmann temp is non-zero.
        #This is already done in the function.
        mn=min(energies)
        energies=[i-mn for i in energies]
        xb=[]
        for j,T in enumerate(args.boltz):
            xb.append(boltz_dist(energies,T))
        #print xb

        if args.EperAtom:      print("\nConfig ID, Rel. Energy/atom [eV], Boltzmann Weight (at %s K)"%(" K ".join([str(x) for x in args.boltz])))
        elif args.Eperfu:    print("\nConfig ID, Rel. Energy/fu [eV], Boltzmann Weight (at %s K)"%(" K ".join([str(x) for x in args.boltz])))
        else:     print("\nConfig ID, Rel. Energy [eV], Boltzmann Weight (at %s )"%("K ".join([str(x) for x in args.boltz])))
        print("%s"%"-"*60)
        str1=""
        for i in range(len(energies)):
            str1+="%-15s %6.3f  "%(  args.inpf[i].replace(".magres","").split("./")[-1],energies[i])
            for j,T in enumerate(args.boltz):
                #xb.append(boltz_dist(energies,T))
                str1+= " %.4f "%xb[j][i]
            str1+="\n"
        print(str1)
    else:
        print("All configurations will be equally weighted in the convoluted NMR spectrum...")
        xb=[[1.0/len(energies) for i in range(len(energies))]] #If no Boltzmann distribution then all configs are weighted equally.

    ################
    # Plotting part#
    ################
    if args.no_plot: exit()

    #Modification in plotting style must be done in the beginning.
    #plt.rc()
    #font = {'family' : 'normal',
    #        'weight' : 'bold',
     #       'size'   : 22}	
    #matplotlib.rc('font', **font)
    #plt.rc('font',**font)

    #plt.rcParams.update({'font.size': 36})
    #plt.rc('axes',titlesize=24,labelsize=24)

    plt.rc('font',size=20) #prev: 20 pt

    ref=args.ref_shift
    pts=2000 #orig:1000 #10k is too slow for SOPRANO., 5k is managable for feq sites, for 17O with many sites too slow.
    if args.atype=='O': pts=800
    gamma=args.gamma #Broadening factor(ppm).

    vals=[]
    for shifts in all_shifts: #This is in the original input file order.
        vals.extend(list(shifts.values()))
        #print shifts
    #vals.sort()

    try:mn=min(vals);    mx=max(vals)
    except: mn=0.;mx=0.
    if args.minmax:
        x0=args.minmax[0]
        xf=args.minmax[1]

        if x0>mn or xf<mx: print("Warning: Chosen range does not cover the whole NMR spectrum. Consider refining the input range to cover (%.1f,%.1f) ppm."%(mn-gamma*2,mx+gamma*2))
    else:
        #dx=abs(mx-mn)*0.1
        #x0,xf=mn-dx,mx+dx
        x0,xf=mn-gamma*2,mx+gamma*2

    if args.separate:
        fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
        
    else:
        fig, ax1 = plt.subplots()

    if args.save: outf3=open("%s-%s.dat"%(stoich,target),'w')
    cc=-1
    cl=['k', 'b',  'r', 'c', 'm', 'y', 'w','g']
    #cl=[ 'g', 'C1','C4','C5','C6','C7','C8', 'c', 'm', 'y', 'k', 'w']
    for Tind,T in enumerate(args.boltz): 
        cc+=1
        print("Temperature: %d K"%T)
        if args.quad:
            print("\n1st and 2nd-order quadrupolar effects are included in the plot.")

            try:
                #from soprano.collection import AtomsCollection
                from soprano.calculate import nmr#,simpson
                from soprano.calculate.nmr import NMRFlags
                from soprano.properties.nmr import *
            except:
                print("SOPRANO package is needed for the quadrupolar effects. (use pip install soprano --user ) ");exit()


            all_ints=[]
            for ind,inpf in enumerate(args.inpf):
                    try:xb[Tind][ind]
                    except: continue
                    if np.isclose(xb[Tind][ind],0.,atol=1e-2): print("%s is skipped due to very low Boltzmann factor..."%inpf);continue #doesn't waste time on the configs with no contribtion to the final spec. #Originally atol=5e-3
                    print(inpf)
                    atoms=ase.io.read(inpf)
                    myNMR=nmr.NMRCalculator(atoms)
                    #myNMR.set_larmor_frequency(larmor_frequency=args.larm,larmor_units='MHz',element=args.atype)
                    myNMR.set_larmor_frequency(larmor_frequency=args.larm,larmor_units='MHz',element="1H")

                    myNMR.set_powder(N=32,mode='hemisphere')#to give multiple orientation to the crystal. Higher N (orientations) the more accurate  (and costly) it is. Not much diff btw sphere and hemisphere. hemisphere is the default.
                    #myNMR.set_single_crystal(theta=180, phi=180) #does not work well.
                    myNMR.set_reference(ref, args.atype)
                    print(myNMR._references)
                    print("Larmor frequency for {0:.3s}: {1:.2f} MHz".format(args.atype,myNMR.get_larmor_frequency(args.atype)))
                    print("Field: {0:.2f} T".format(myNMR.B))
                    #print "Powder orientations: {0}".format(len(myNMR._orients[0]))

                    labels = atoms.get_array('labels')
                    indices = atoms.get_array('indices')
                    # One can also recreate the Jmol-style labels if needed
                    jmol_labels = ["{0}_{1}".format(l, i) for l, i in zip(labels, indices)]
                    #print "The labels are {0}".format(', '.join(jmol_labels))

                    # Isotropy, Anisotropy and Asymmetry (Haeberlen convention)
                    iso = MSIsotropy.get(atoms)
                    aniso = MSAnisotropy.get(atoms)
                    asymm = MSAsymmetry.get(atoms)

                    vzz = EFGVzz.get(atoms)
                    #qP = EFGQuadrupolarConstant(isotopes={'H': 2}) # Deuterated; for the others use the def.
                    qP = EFGQuadrupolarConstant()
                    qC = qP(atoms)/1e6 # To MHz
                    #EFGaniso=EFGAnisotropy.get(atoms)
                    EFGasymm=EFGAsymmetry.get(atoms)

                    print('Label\tShield\tIsotropy\tAnisotropy\tAsymmetry\tVzz\t\tCQ/Chi\t\teta_Q')
                    for i, jl in enumerate(jmol_labels):
                        if args.atype in jl:  print('{0}\t{7:.2f}\t{1:.2f} ppm\t{2:.2f} ppm\t{3:.2f}\t\t{4:5.2f} au\t{5:5.2f} MHz\t{6:5.2f}'.format(jl, ref-iso[i], aniso[i], asymm[i],vzz[i], qC[i],EFGasymm[i],iso[i]))

                    #ints,freqs=myNMR.spectrum_1d(args.atype, min_freq=(ref-xf), max_freq=(ref-x0), bins=pts,freq_broad=gamma, freq_units='ppm', effects=45)#NMRFlags.MAS )#contains CS_ISO+Q_2_SHIFT+Q_2_ORIENT_MAS) MAS=41 by def, total=47 could be better incl. CS_ORIENT and Q_1_ORIENT as well. #45 matches well with other programs. 33 is also good #THIS STILL NEEDS TESTING; #Always been using the 45 option, best  match with WSolids for 27Al and 71Ga.
                    #myNMR.set_reference(0.0, args.atype)
                    ints,freqs=myNMR.spectrum_1d(args.atype, min_freq=(ref-xf), max_freq=(ref-x0), bins=pts,freq_broad=gamma, freq_units='ppm', effects=45,use_reference=0,use_central=0)

                    wints=np.array([ xb[Tind][ind]*x for x in ints])#weighted by the Boltzmann factors of each config
                    if len(all_ints)==0:all_ints=wints
                    else:all_ints+=wints

                    if args.overlay and Tind==len(xb)-1: #only do for the last Temp. step.
                        label=inpf.split("/")[-1].split('.')[0]
                        if args.mono:color='k'
                        else:color=None
                        ax1.plot(ref-freqs,wints, label=label,ls="dashed",lw=3,color=color)
                        #ax1.plot(freqs,wints, label=label,ls="dashed",lw=3,color=color)
                        if args.save: 
                            outf4=open('%s'%(inpf.split('.')[0]+'.dat'),'w')
                            for j in range(len(wints)):   outf4.write("%14.5f %14.5f\n"%(ref-freqs[j],wints[j]))
                            outf4.close()
            if T==0:label="Total MAS"
            else:label='%d K'%T
            try:ax1.plot([ref-f for f in freqs],all_ints, label=label,lw=3,color=cl[cc%len(cl)])#,color='black')
            except:continue
            #ax1.plot([f for f in freqs],all_ints, label=label,lw=3,color=cl[cc%len(cl)])#,color='black')

            if args.save: 
                for i,ints in enumerate(all_ints):  outf3.write("%15.5f %15.7f\n"%((ref-freqs)[i],ints))


            if args.show_lines:
                if args.color_lines:#coloring wrt coord no
                    for i in range(len(all_shifts)):
                        shifts=all_shifts[i]
                        for atom in list(shifts.keys()): 
                        #print atom
                            if crds[atom]==4: clr="red"
                            elif crds[atom]==5:clr="green"
                            elif crds[atom]==6:clr="blue"
                            else:clr='black'
                            ax1.vlines(shifts[atom],0,max(all_dists),color=clr,linestyles='solid',label=None,alpha=0.5) #label="crd: %d"%crds[atom]

                else:
                    for i in range(len(all_shifts)):
                        shifts=all_shifts[i]
                        ax1.vlines(list(shifts.values()),0,max(all_dists),color='black',linestyles='solid',label=None,alpha=0.4)

        else:
	    #TODO add the Temperature loop here !!!
            path=np.linspace(x0,xf,pts)
            line=np.zeros((pts))
            line2=np.zeros((pts))
            #for val in vals:
            for i in range(len(all_shifts)): #loop over input files
		#xb[Tind][ind]
                if np.isclose(xb[0][i],0.,atol=1e-2): print("%s is skipped due to very low Boltzmann factor..."%args.inpf[i]);continue #doesn't waste time on the configs with no contribtion to the final spec. #Originally atol=5e-3

                shifts=all_shifts[i]
                cline=np.zeros((pts))
                reduced=getReduced(ase.io.read(args.inpf[i]))[0]
                for val in list(shifts.values()):
                    for j in range(pts):
                        x=path[j]
                        #if args.bType == 'Lor'
                        lor=lorentzian(x,val,gamma)*xb[Tind][i]
                        gau=gaussian(x,val,gamma)*xb[Tind][i]
                        line[j]+=lor
                        line2[j]+=gau

                        if args.bType=='Lor':                     cline[j]+=lor
                        elif args.bType=='Gau':                     cline[j]+=gau

                if args.overlay: 
                    label=args.inpf[i].split("/")[-1].split('.')[0]
                    label=reduced+' - '+ SGs[i] #stoich
                    if args.mono:color='k'
                    else:color=None
                    ax1.plot(path,cline, label=label,ls="dashed",lw=3,color=color)
                    if args.save: 
                        outf4=open('%s'%(args.inpf[i].split('/')[-1].split('.')[0]+'.dat'),'w')
                        for j in range(len(cline)):   outf4.write("%14.5f %14.5f\n"%(path[j],cline[j]))
                        outf4.close()
                    #elif args.bType=='Gau': ax1.plot(path,gaussian(x,val,gamma)*xb[i], label=inpf.split("/")[-1].split('.')[0],ls="dashed")
            if args.save: 
               if args.bType=='Lor':     ln= line
               elif args.bType=='Gau':   ln=line2

               for i,ints in enumerate(ln):  outf3.write("%14.5f %14.5f\n"%(path[i],ints))

            if args.show_lines:
                if args.color_lines:#coloring wrt coord no
                    for i in range(len(all_shifts)):
                        shifts=all_shifts[i]
                        for atom in list(shifts.keys()): 
                        #print atom
                            if crds[atom]==4: clr="red"
                            elif crds[atom]==5:clr="green"
                            elif crds[atom]==6:clr="blue"
                            else:clr='black'
                            ax1.vlines(shifts[atom],0,min(max(line),max(line2))*xb[Tind][i],color=clr,linestyles='solid',label=None,alpha=0.5) #label="crd: %d"%crds[atom]

                else:
                    for i in range(len(all_shifts)):
                        shifts=all_shifts[i]
                        ax1.vlines(list(shifts.values()),0,min(max(line),max(line2))*xb[Tind][i],color='black',linestyles='solid',label=None,alpha=0.4)

            if not args.color_lines:
                if args.bType == 'Lor':
                    ax1.plot(path,line,color='black', label='Lorentzian',lw=3)
                elif args.bType == 'Gau':
                    ax1.plot(path,line2,color='black', label='Gaussian',lw=3)
                else:
                    ax1.plot(path,line,color='red', label='Lorentzian',lw=3)
                    ax1.plot(path,line2,color='blue', label='Gaussian',lw=3)

    #end of args.inpf (input files) loop.
    if args.save: outf3.close()

    plt.xlim(xf,x0)
    plt.ylim(0)
    if args.ref_shift != 0:
        if args.quad_shift:
            plt.xlabel('Chemical shift w/ quadrupolar corr. [ppm] (%s= %.1f ppm)'%(r'$\sigma_{ref}$',ref_shift))
        else:
            plt.xlabel('Chemical shift [ppm] (%s= %.1f ppm)'%(r'$\sigma_{ref}$',ref_shift))
            plt.xlabel('Chemical shift (ppm)')
    else:                    
        if args.quad_shift:
            plt.xlabel("Isotropic shielding w/ quadrupolar corr. [ppm]")
        else:
            plt.xlabel("Isotropic shielding [ppm]")

    #plt.legend()
    
    #Read in the reference exp. data.
    if args.ref_data:
        print()
        if not args.separate:
            ax2=ax1.twinx()
        else: None #defined earlier
            
        ax2.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            left='off',
            right='off',
            labelbottom='off') # labels along the bottom edge are off
        ax1.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            right='off',         # ticks along the top edge are off
            labelleft='off') # labels along the bottom edge are off
        

        cc=-1
        #cl=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        cl=[ 'g', 'C1','b','r','C4','C5','C6','C7','C8', 'c', 'm', 'y', 'k', 'w'] #C1:orange
        #shift_ref=args.shift_ref; max_y=0.
        for ref_file in args.ref_data:
            cc+=1
            print("Reading reference spectrum data from %s."%ref_file, end=' ')
            stdout.flush()
            ref=[[],[]]
            cnt=0
            for ln in open(ref_file,'r'):
                cnt += 1
                x=ln.split()
                if x[0]=="#": continue
                elif cnt%args.skip_ref==0:#for skipping some data points.
                    ref[0].append(float(x[0]))
                    ref[1].append(float(x[1]))

            ref[1]=np.array(ref[1])/max(ref[1])
            ref[1]-=min(ref[1]) #shift to 0.
            print("Done...")

            label=ref_file.split("/")[-1]
            try:label=reduced #stoich
            except:None
            ax2.plot(ref[0],ref[1],color=cl[cc%len(cl)],label=label,lw=3)
        #if shift_ref != 0.: plt.ylim(0,max_y)

    if not args.no_legend: # and args.overlay:
        ax1.legend(fancybox=True, shadow=True, prop={'size': 18},loc='upper right') 

    #Hiding y-axis title
    frame1 = fig.gca()
    frame1.axes.get_yaxis().set_visible(False)

    if args.separate:
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.0)

    plt.show()

    exit()


#################
# Deleted Parts #
#################

# {'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'}
# {'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'} 

    #Example For computing NQR transition freqs.
    freqs = []

    for atom in atoms.species('O'):
        print((atom, "S={:.1f} Q={:.2f} millibarn".format(atom.spin, atom.Q)))

        for m in available_m(atom.spin):
            freq_MHz = calc_nqr(atom, m) / megahertz
            print(("  m={:.1f}->{:.1f} freq = {:.3f} MHz".format(m, m+1, freq_MHz)))

            freqs.append(freq_MHz)

    print(("Mean freq = {:.3f} Mhz".format(np.mean(freqs))))

    """
            #Shift the ref spectrum to 0 (along y-axis). This is done automatically by having twinx.
            #mn=min(ref[1])
            #ref[1]=[i-mn for i in ref[1]]
            if shift_ref != 0.: #This does not work properly, as the ranges do not compily.
                sh=max(ref[1])*shift_ref/100
                tmp=[i+sh for i in ref[1]]
                ref[1]=deepcopy(tmp)
                mx=max(ref[1])
                if mx>max_y:max_y=mx
                print max_y
    """
        #ax2.set_xticklabels([])
        #ax2.set_yticklabels([])

    #frame1.axes.yaxis.set_ticklabels([])
    #frame2 = ax2.gca()
    #frame2.axes.get_yaxis().set_visible(False)

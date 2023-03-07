#!/usr/bin/env python3

import argparse,os

import matplotlib.pyplot as plt
import numpy as np


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


def lorentzian(w, w0, gamma):
    return (1/np.pi)*gamma/((w-w0)**2+gamma**2)

def gaussian(w, w0, gamma):
    return ((np.sqrt(np.log(2)/np.pi))/gamma)*np.exp(-np.log(2)*((w-w0)/gamma)**2)
    sigma = alpha / np.sqrt(2 * np.log(2))

    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-inpf","--inpf", required=1, nargs="*", type=str,
    help="Input data file(s) in OUTCAR format.")

    parser.add_argument("-ref","--ref", required=0, nargs="*", type=str,
    help="Reference data file(s) in data format (e.g. energy , intensity).")

    parser.add_argument("-sh","--shift", required=0, nargs=1, type=float, default=0.0,
    help="Shift applied to input data (not to the reference). Def= 0.0 eV.")

    parser.add_argument("-lim","--xlimits", required=0, nargs=2, type=float, default=None,
    help="X-axis plotting range. Def= Min-max values from the absorption energies are used")


    parser.add_argument("-btype","--btype", required=0, type=str, default=None,choices=['Lor','Gau','L','G'],
    help="Apply broadening to the input data. Def: no")

    parser.add_argument("-gam","--gamma", required=0, nargs=1, type=float, default=1.5,
    help="Broadening factor (Gamma), def: 1.5")

    parser.add_argument("-bpts","--bpoints", required=0, nargs=1, type=int, default=1000,
    help="Number of points to use in broadening, def: 1,000")

    parser.add_argument("-boltz","--boltz", type=float, nargs="*",default=[0.], help="To weigh the NMR peaks from multiple configurations using Boltzmann distribution at a given temperature [K] based on their formation energies to get a convoluted NMR spectrum. Def: no weighting is done. Usage: -boltz TEMP")
    parser.add_argument("-ca","--coord_anlys", action='store_true', default=False, help="Perform detailed cordination distortion analysis for the chosen atom type.")
    parser.add_argument("-cna","--cna", action='store_true', default=False, help="Perform detailed closest neighbour analysis for the chosen atom type within the --max_dist Angstroem.")
    parser.add_argument("-bvs_B","--bvs_B", type=float, default=0.37, help="Emprical constant, B value for computing the Bond Valance Sum (BVS) for the selected atom types. Def: 0.37 A ")
    parser.add_argument("-bvs_R0","--bvs_R0", type=float, help="Bond valance parameter, R0 value for computing BVS for the selected atom types. No default: if no value given then --bvs_file will be used to obtain R0 values. ")
    parser.add_argument("-bvs_file","--bvs_file", type=str, default=os.environ['HOME']+"/bvparm2013.cif", help="File containing R0 values for computing BVS for the selected atom types (formatting taken from VESTA code pack). If  multiple enetries for given caton/anion pair, then the first entry will be used. Def: ~/bvparm2013.cif")
    parser.add_argument("-bvs_ox","--bvs_ox", type=int, help="Oxidation state for the central atom (cation). If no input given the first available state is used by default. ")
    parser.add_argument("-sp","--save_pol", action='store_true', default=False, help="Save all polyhedra from the collection of input files into a single file (out.res) for visualisation purposes. Def: False")


    args = parser.parse_args()


    # general options for plot
    font = {'family': 'serif', 'size': 18}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(11.69, 8.27))
    #fig = plt.figure()
    #gs = GridSpec(2, 1)#, width_ratios=[1, 2],height_ratios=[1, 1])
    #ax1 = plt.subplot(gs[0])  #Top subplot
    #ax2 = plt.subplot(gs[1])#   , sharex=ax1)

    for i,inp in enumerate(args.inpf):
        print("Reading %s ..." %inp)
        #np.load(inp)
        try:
          with open(inp, "r") as f:
            flag=0
            data=[]
            for ln in f.readlines():  
                if "frequency dependent IMAGINARY DIELECTRIC FUNCTION" in ln:
                    flag=1
                elif flag:
                    if not ln=="": 
                        x=ln.split()
                        try:data.append([float(x[0]),sum([float(a) for a in x[1:]])])                  
                        except:None #print(x)
                        
                    else: flag=0;break

                    if 'frequency dependent      REAL' in ln:
                        flag=0;break


          data=np.array(data).T
        except Exception as err: print("Can't read the file:", err);continue

        if len(data)==0: continue
        
        if args.btype: #Gaussian/Loretzian broadening of the data
            print('Broadening as requested')
            if args.xlimits:
                x0=args.xlimits[0];xf=args.xlimits[1]
            else:
                x0=np.min(data[0])+args.shift;xf=np.max(data[0])+args.shift

            pts=10000 #10 k gives good sampling.
            path=np.linspace(x0,xf,pts)
            line=np.zeros((pts))
            gamma=args.gamma #For gamma <0.05, one needs more data points, otherwise the peak height is overemphasized!!!

            #TODO:add the Boltzmann weight and multiple Temp    
            for i in range(len(data[0])):     
                if data[1][i]==0.0: continue #speeds up significantly withut data loss
                for j in range(pts): #A cutoff can be added based on the val
                    x=path[j]
                    val=data[0][i]+args.shift #correct
                    if args.btype=='Lor':
                        pt=lorentzian(x,val,gamma)*data[1][i]#/100#*xb[Tind][i]  
                    elif args.btype=='Gau':
                        pt=gaussian(x,val,gamma)*data[1][i]#*xb[Tind][i]
                    else:pt=0

                    line[j]+=pt

                    
            maxI=max(line)
            if 1: plt.plot(path,line/maxI,label=inp.split('/')[0]) #label='%s - %s'%(sg,inpf.split('/')[-1].split('.')[0]),color=clr) #Use Lorentzian
        
        else: #No broadening of the data
            plt.plot(data[0]+args.shift,data[1]/np.max(data[1]),label=inp.split('/')[0])
        
        if 0: np.savetxt('%s.dat'%(i+1),data.T)


    if args.ref: #Read reference data
        try:
          for i,ref in enumerate(args.ref):
            print("Reading %s ..." %ref)
            data=np.loadtxt(ref,skiprows=1,delimiter=',').T
            plt.plot(data[0],data[1]/np.max(data[1]),label='Ref #%d'%(i+1),ls='--',lw=2.)
        except Exception as err: print("Can't read the file:", err)


    plt.xlim(args.xlimits) #;plt.ylim(args.xlim[1])
    plt.xlabel('Absorption energy / eV')
    plt.ylabel('Intesity / a.u.')
    plt.legend()
    plt.show()


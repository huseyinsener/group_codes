#!/usr/bin/env python3

import argparse,os

import matplotlib.pyplot as plt
import numpy as np

def lorentzian(w, w0, gamma):
    return (1/np.pi)*gamma/((w-w0)**2+gamma**2)

def gaussian(w, w0, gamma):
    return ((np.sqrt(np.log(2)/np.pi))/gamma)*np.exp(-np.log(2)*((w-w0)/gamma)**2)
    sigma = alpha / np.sqrt(2 * np.log(2))

    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)


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
    for i,ref in enumerate(args.ref):
        print("Reading %s ..." %ref)
        data=np.loadtxt(ref,skiprows=1,delimiter=',').T
        plt.plot(data[0],data[1]/np.max(data[1]),label='Reference #%d'%(i+1))


plt.xlim(args.xlimits) #;plt.ylim(args.xlim[1])
plt.xlabel('Absorption energy / eV')
plt.ylabel('Intesity / a.u.')
plt.legend()
plt.show()


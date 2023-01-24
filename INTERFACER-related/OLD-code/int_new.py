#!/usr/bin/env python3
from scipy.optimize import minimize
import numpy as np
import argparse
import ase.io
import ase
from ase.visualize import view
from os import system
from sys import exit
import fractions
import ase.build
import copy
import time
import ase.build.tools
import matplotlib.pyplot as plt
import os
import scipy.linalg as la
from math import ceil
import spglib
import re
import sys

	

class Cell():
	"""user defined class for a unit cell"""
	def __init__(self,ase_atoms):
		#self.cell[i] returns ai vector
		self.cell = ase_atoms.cell
		self.symbols = ase_atoms.get_chemical_symbols()
		self.positions = [x for x in ase_atoms.get_positions()]
		self.set_props()
		self.pbc = [True, True, True]
		self.frac = [[0,0,0] for x in range(0,len(self.positions))]
		self.all_frac_from_Cart()
	
	def shift(self,tcart):
		"""shift all positions by tcart and then wrap alll"""	
		for i in range(0, len(self.positions)):
			self.positions[i] += tcart
		self.all_frac_from_Cart(wrap=True)

	def crep(self, crt):
		#repeats the slab cr times along the c direction
		if crt is not None and crt != 1:
			new_top_cart = copy.deepcopy(self.cell)
			new_top_cart[2] = crt*new_top_cart[2] 
			self.convert_to(new_top_cart,0)
	

	def add_vac(self,vac):
		#adds vac to top and bottom of the slab along the a3 direction
		n = self.cell[2]/np.linalg.norm(self.cell[2])		
	
		#add 2n to a3
		self.cell[2] += 2 * vac * n
		#print(self.cell[2])
	
		#shift the positions by vac *n and then update all the fractional positions
		for i in range(0,len(self.positions)):
			self.positions[i] += n*vac

		self.all_frac_from_Cart(wrap=False)		


	def square(self):
		self.all_frac_from_Cart(wrap=True)	
		
		"""
		print(48)	
		for i,f in enumerate(self.frac):
			print(f, self.positions[i])
		print(49)
		"""
	
		#squares the cell
		n = np.cross(self.cell[0],self.cell[1])
		n = n/np.linalg.norm(n) 
		len_a3 = np.dot(self.cell[2],n)
		self.cell[2] = len_a3*n		
		self.pbc[2] = False
			
		#update the fractional coordinates from the cartesian ones
		self.all_frac_from_Cart(wrap=True)	#can wrap as have removed pbc in the a3 direction

		"""
		print(64)
		for i,f in enumerate(self.frac):
			print(f, self.positions[i])
		print(67)
		"""		


	def all_Cart_from_frac(self):
		#sets all of the Cart positions from frac
		for i,f in enumerate(self.frac):
			p = self.frac_to_Cart(f)
			self.positions[i] = p
	

	def wrap_frac(self, frac, tol=10**-5):
		#wraps coordinates to lie in range[0,1-tol]
		frac2 = []
		
		for i,f in enumerate(frac):
			f2 = f
			if self.pbc[i]:
				f2 = f2%1
				if f2 >= 1-tol:
					f2 = 0
			frac2.append(f2)

		return frac2
	
	def all_frac_from_Cart(self, wrap=True):
		#sets all of the fractional positions from Cart
		#if wrap is true it wraps the fractional coordinates, subject to pbc and resets the Cartesians from the wrapped fracs

		for i,p in enumerate(self.positions):
			f = self.Cart_to_frac(p)
			if wrap:
				f = self.wrap_frac(f)
			self.frac[i] = f		
			
		#if wrap is true update all Cartesians from fractionals
		if wrap:
			self.all_Cart_from_frac()
			

	def set_props(self):
		self.atom_count = len(self.symbols)
		self.volume = np.linalg.det(self.cell)
	
	def Cart_to_frac(self, Cart):
		#converts a cartesion vector to a fractional one
		M = np.matrix(copy.deepcopy(self.cell)).T
		M = M.I
		return np.dot(np.array(M),Cart)

	def frac_to_Cart(self,frac):
		#converts a fractional vector into a cartesian one
		M = np.matrix(copy.deepcopy(self.cell)).T
		return np.dot(np.array(M),frac)
	
	def add_atom(self, sym, frac=None, Cart=None):
		#adds atom at the specified fractional coordinates
		self.symbols.append(sym)
		if frac is not None:
			self.frac.append(frac)
			self.positions.append(self.frac_to_Cart(frac))		
		elif Cart is not None:
			self.positions.append(Cart)
			self.frac.append(self.Cart_to_frac(Cart))
			
	def convert_to(self, new_Cart, origin):
		#change to supercell with new_Cart as the basis vectors 
		#need to check repeats of atoms to see if they lie in the new cell
		
		#update unit vectors
		prim_Cart = self.cell
		self.cell = new_Cart

		M = np.transpose(prim_Cart)
		M = np.linalg.inv(M)
		A = np.transpose(self.cell)
		

		#subtract origin from coordinates of all atoms
		for i,p in enumerate(self.positions):
			self.positions[i] = np.array(p) - origin
		
		#now atoms are in the correct place relative to new axes
		#wrap them inside the cell
		self.frac = []
		for i,p in enumerate(self.positions):
			f = self.Cart_to_frac(p)
			fw = np.array([x % 1 for x in f])
			
			pw = self.frac_to_Cart(fw)
			self.positions[i] = pw	
			self.frac.append(fw)
	
	
		
		#now loop over atoms shifting by 27 prim_Cart options, convert to frac
		#if not already in list then add atoms to cell	
		#create shifts to adjacant primitives
		shifts = []
		for i in range(-1,2):	
			for j in range(-1,2):
				for k in range(-1,2):
					temp = i*prim_Cart[0]+j*prim_Cart[1]+k*prim_Cart[2]
					shifts.append(temp) 
		 
		#try to shift each atom by all adjacant primitives
		#if any new atom is added reapeat the process
		atom_added = True
		while atom_added:
			atom_added = False
			for i,p in enumerate(self.positions):
				sym = self.symbols[i]
				for s in shifts:
					p2 = p + s
				
					f2 = self.Cart_to_frac(p2)	
					f2 = np.array([x % 1 for x in f2])

					#check if f2 is in the fractional coords
					duplicate = False
					for fc in self.frac:
						tol = 10**-6
						same = True
						for i in range(0,3):
							dif = abs(f2[i]-fc[i])
							if dif > tol and abs(dif-1) > tol:
								same = False
								
						if same:	
							duplicate = True	
  
					if not duplicate:
						atom_added = True
						self.add_atom(sym, frac=f2) 
					
				
		#should now have added all atoms	
		self.set_props()		
	
	

		
		

def lcm(a,b):
	return (a*b)//fractions.gcd(a,b)

def get_cut_vectors(miller, atoms_cell):
	"""returns d,e,f in fractional coordinates relative to initial cell and origin in cartestian coordintes"""
	#from miller indicies construct 2 lattice vectors in the required plane
	h,k,l = miller
	a = atoms_cell[0]
	b = atoms_cell[1]
	c = atoms_cell[2]
	
	zero_inds = np.argwhere(np.array(miller)==0)

	#no zero miler indicies
	if len(zero_inds) == 0:
			lm = lcm(h,l);	d = (lm/h,0,-lm/l)		#vector from c intersection to a
			lm = lcm(k,l);	e = (0,lm/k,-lm/l)		#vector from c` intersection to b
			#pick f so that cell is left handed
			f = (0,0,1)
			origin = c*1.0/l

	elif len(zero_inds)==3:
			print("miller not a valid plane")
			raise ValueError

	#one zero milller index
	elif len(zero_inds)==1:
			#set d to be the axis that is in the plane
			d = [0,0,0]
			d[int(zero_inds[0])] = 1

			#construct e from the other 2 vectors
			s = set((0,1,2))
			s.remove(int(zero_inds[0]))
			#print("s is",s)
			H = s.pop()
			K = s.pop()
			lm = lcm(miller[H],miller[K])
			#print(H,miller[H],K,miller[K],lm)
			e = [0,0,0]
			e[H] = lm/miller[H]
			e[K] = -lm/miller[K]
			#print("e is", e)

			#pick f - could be improved...
			f = [0,0,0]
			f[H] = 1

			#pick origin
			origin = 1.0/miller[H] * atoms_cell[H]


	#two zero miller indicies
	elif len(zero_inds)==2:
			d = [0,0,0]
			#print zero_inds
			d[zero_inds[0][0]] = 1
			e = [0,0,0]
			e[zero_inds[1][0]] = 1
			f_ind = np.argwhere(np.array(miller)!=0)[0][0]
			f = [0,0,0]
			f[f_ind] = 1
			origin = atoms_cell[f_ind]*1.0/miller[f_ind]
			#print d,e,f
			#print origin

	#consistently want the surface chosen (defined by d and e)
	#to end up lying on the bottom of the slab that is made
	#don't worry about handedness here, sort that out later
	#To do this want f dot origin to be negative then a3 will become squared f

	fc = f[0]*a + f[1]*b + f[2]*c
	if np.dot(fc,origin) > 0:
			f = -np.array(f)

	return [d,e,f],origin

class Slab():
	"""slab class to keep things neat"""
	def __init__(self,miller, Cell):
		self.miller = miller
		self.Cell = Cell
		self.frac, self.origin = get_cut_vectors(self.miller,self.Cell.cell)
		self.Cart_from_frac()		#converts fractional vecs into cartesians
		self.Gauss_Reduce()		#changes to Gauss reduced basis
		self.volume = abs(np.linalg.det(self.Cart))
		self.set_atom_count()		
				

	def set_Rmax(self,nmax):
		"""sets maximum distance that needs to be considered from
		the max atoms per slab option"""
		#could perform gauss reduction on any match between slabs
		#so any match has to be mappable to a match where the angles
		#are between 60 and 150 degrees
		#nmax is max number of atoms allowed in the slab
		dmin = np.linalg.norm(self.Cart[0])
		Area = np.linalg.norm(np.cross(self.Cart[0],self.Cart[1]))
		Amax = float(nmax)/self.atom_count * Area	
		#Area > |a1||a2| |sin(theta)|, 60 <= theta <= 150
		Rmax = Amax/dmin * np.sin(60.0/180 * np.pi)
		self.Rmax = Rmax
		self.nmax = nmax	
		#also set max_repeats
		self.max_repeats = np.floor(nmax/self.atom_count)
				
	def Gauss_Reduce(self):
		#perform Gauss reduction to find the 2 shortest vectors
		#u and v are three vectors but that doesn't change a damn thing
		u = self.Cart[0]
		v = self.Cart[1]
		if np.linalg.norm(v) < np.linalg.norm(u):
			t = u;	u = v;	v = t
		
		def reduced(u,v):
			#checks if basis is reduced
			check = True
			if np.linalg.norm(v) < np.linalg.norm(u):
				check = False
			if 2 * np.dot(u,v) > np.dot(u,u):
				check = False
			return check
	
		i = 0
		while not reduced(u,v) and i < 1000:
			i += 1
			#v -> v - mu , m interger chosen to minimise v
			m = np.ceil(2 * np.dot(v,u)/np.dot(u,u))
			v1 = v - m* u
			v2 = v - (m-1)* u
			if np.linalg.norm(v1) <= np.linalg.norm(v2):
				v = v1
			else:
				v = v2
			
			if np.linalg.norm(v) < np.linalg.norm(u):
				t = u;	u = v;	v = t
		
		if not reduced(u,v):
			print("reduction algorithm has failed :(")
			exit()	
		
		#set the cartesian slab vecs		
		self.Cart[0] = u
		self.Cart[1] = v	
		self.frac_from_Cart()					
				
	def Cart_from_frac(self):
		#set the cartesian slab vectors from fractional coordintaes
		#in terms of the vectors of the original cell parsed
		self.Cart = []
		for f in self.frac:
			v = 0
			for i,comp in enumerate(f):
				v += comp*self.Cell.cell[i]
			self.Cart.append(v)

	def frac_from_Cart(self):
		#set the fractional vectors in terms of the inital cell vectors
		#miller index notation for a1 and a2 in terms of inital cell
		#useful for keeping track of which directions are involved
		
		M = np.mat(self.Cell.cell).T
		C = np.mat(self.Cart).T
		f = np.dot(M.I,C)
	
		
		self.frac[0] = np.array(f[:,0])
		self.frac[0] = np.array(int(x) for x in self.frac[0])
		self.frac[1] = np.array(f[:,1])
		self.frac[1] = np.array(int(x) for x in self.frac[1])
			
	def set_atom_count(self):
		#sets the number of atoms in the slab
		n = int(round(self.volume/self.Cell.volume))
		self.atom_count = n * self.Cell.atom_count
		
def find_vecs(slab,Rmax):
	"""find all vectors with length <= Rmax
	only returns 1 vec in each direction, e.g. will not contain [0,1] and [0,2] 
	or [0,1] and [0,-1]"""
	u = slab.Cart[0]
	v = slab.Cart[1]
	
	xlim = Rmax/(np.linalg.norm(u-np.dot(u,v)/np.dot(v,v)))
	xlim = int(np.floor(xlim))	
	
	vecs = []
	
	for x in range(-xlim,xlim+1):
		#loops over the range of x vectors where its possible for length to be <= Rmax
		A = np.dot(v,v)
		B = 2 * x * np.dot(u,v)
		C = x**2 * np.dot(u,u) - Rmax**2
		ymax = int(np.ceil((-B + np.sqrt(B**2 - 4 * A * C))/(2*A)))
		ymin = int(np.floor((-B - np.sqrt(B**2 - 4 * A * C))/(2*A)))
		for y in range(ymin,ymax+1):
			#loops over the range of y where its possible for length to be <=Rmax
			vec = Vec(u,v,x,y)					
			tol = 0.0001	#length not zero and vecs are parallel
			if vec.length <= Rmax and vec.length > tol:
			#check for parallel vecs in v:
				par = False
				for i,v2 in enumerate(vecs):
					if np.linalg.norm(np.cross(v2.Cart,vec.Cart)) < tol:	
						par = True	
						if v2.length < vec.length:
							continue
						else:
							vecs[i] = copy.copy(vec)
				if not par:
					vecs.append(vec)
				
	return vecs

class Vec():
	"""class to contain infomation about a vector choice"""
	def __init__(self,u,v,x,y):
		#self.frac are in terms of the slab vectors
		#so ARE NOT in terms of the original cell vectors
		#generally need to convert
		self.frac = np.array([x,y])		#<-- don't need to keep track of this, find at end by expressing in inital cell Basis
		self.Cart = x * u + y * v
		self.length = np.linalg.norm(self.Cart)
	
def build_pairs(vecs, slab):
	"""loop over vecs and construct pairs"""
	pairs = []
	a3 = slab.Cart[2]
	for i,v1 in enumerate(vecs):
		for v2 in vecs[i+1:]:
			#add all 4 possibilites
			for j in range(0,4):
				pairs.append(Pair(v1,v2,slab,j))	

	#set the atom_counts
	pairs2 = []
	for p in pairs:
		p.set_atom_count(slab)
		if p.atom_count <= p.nmax:
			pairs2.append(p)
	return pairs2
		
class Pair():
	"""pair of vectors"""
	def __init__(self,v1,v2,slab,i):
		#i =0,1,2,3 determines which +-/+- combination to take as the pair
		self.slab = slab
		self.v1 = v1
		self.v2 = v2
		self.i = i

		a3 = slab.Cart[2]
		s1 = (-1)**(i%2)
		s2 = (-1)**(i//2)
		self.frac = [s1*np.array(v1.frac), s2*np.array(v2.frac)]
		self.Cart = np.array([s1*v1.Cart, s2*v2.Cart,a3])
		self.RH()
		self.set_unit_normal()
		self.set_angle()
		self.nmax = slab.nmax
		self.set_atom_count(slab)
		self.set_max_repeats()

	def repeat_slab(self,reps):
		#repeats the slabs by the ints in reps[0] and reps[1]
		self.Cart[0] *= reps[0];	self.frac[0] *= reps[0]
		self.Cart[1] *= reps[1];	self.frac[1] *= reps[1]
		self.atom_count *= reps[0]*reps[1]
		self.set_max_repeats()
		
	def set_max_repeats(self):
		#sets the maximum repeats
		self.max_repeats = np.floor(self.nmax/self.atom_count)	

	def set_angle(self):
		#sets the angle between a1 and a2
		a = self.Cart[0]/np.linalg.norm(self.Cart[0])
		b = self.Cart[1]/np.linalg.norm(self.Cart[1])
		self.angle = np.arccos(np.dot(a,b))
		self.tan_angle = np.tan(self.angle)
		
	def set_unit_normal(self):
		#set the unit normal vector
		#because of right handed convention a3.normal > 0
		n = np.cross(self.Cart[0],self.Cart[1])
		self.unit_norm = n/np.linalg.norm(n)	
				
	def set_atom_count(self,slab):
		#set number of atoms for single unit
		Vol = np.linalg.det(self.Cart)
		self.atom_count = slab.atom_count * Vol/slab.volume
		
		
	def RH(self):
		#make choices right handed
		if np.dot(self.Cart[2],np.cross(self.Cart[0],self.Cart[1])) < 0:
			t = copy.copy(self.Cart[0])
			self.Cart[0] = copy.copy(self.Cart[1])
			self.Cart[1] = copy.copy(t)
		

class Match():
	"""class for matched pairs"""
	def __init__(self, pair1, pair2, max_eig):
		self.pair1 = pair1
		self.pair2 = pair2
		self.max_eig = max_eig
		self.atom_count = int(round(pair1.atom_count + pair2.atom_count))

def match_pairs(pairs1, pairs2, ptol, Rmax):
	"""find the pairs that match. tol is a percentage tolerance on the eigenvalues"""
	#loop over pairs, testing them and creating a list of matches within tolerance
		
	def polar_decomp(p1, p2):
		#decomposes transformation matrix from slab1 to slab2 
		#finds max eignvalue of the stretch
		M1 = np.matrix([p1.Cart[0], p1.Cart[1], p1.unit_norm]).T	
		M2 = np.matrix([p2.Cart[0], p2.Cart[1], p2.unit_norm]).T
		A = np.dot(M2,M1.I)
		P = la.sqrtm(np.dot(A.T,A)) 
	
		eigs, eig_vecs = np.linalg.eig(P)
		p_eigs = [x if x >=1 else 1.0/x for x in eigs]
		return	max(p_eigs)
	
	def length_match(p1, p2, ptol, Rmax):
		#see if it is possible to find multiples of p1 and p2 that make them equal within ptol
		#the stretch can't be more than ptol in any direction (elipse) so this is required
		#take smaller of Rmax and max repeats for each direction

		#for each cell find the max number of repeats to consider in each direction
		#n0 ,n1 for p1 and m0 m1 for p2
		n0 = p1.max_repeats
		t = int(np.floor(Rmax/np.linalg.norm(p1.Cart[0])))
		if t < n0:
			n0 = t	
		t = int(np.floor(Rmax/np.linalg.norm(p1.Cart[1])))
		if t < n0:
			n1 = t
		else:
			n1 = n0
		
		m0 = p2.max_repeats
		t = int(np.floor(Rmax/np.linalg.norm(p2.Cart[0])))
		if t < m0:
			m0 = t	
		t = int(np.floor(Rmax/np.linalg.norm(p2.Cart[1])))
		if t < m0:
			m1 = t
		else:
			m1 = m0

		#that was long but now have max number of repeats to consider for each cell
		#function to match the lengths within ptol
		def lcm_float(a,b,ptol,amax,bmax):
			#a and b are floats, find lcm within ptol
			#amax is max repeats of a, similar for bmax
			i = 1;	j = 1;
			def test(a, b, I, J, ptol):
				A = a*I
				B = b*J
				if 200*abs(A-B)/(A+B) <= ptol:
					return True
				else:
					return False
			while not test(a,b,i,j,ptol) and i <= amax and j <= bmax:
				if i*a < j*b:
					i += 1
				else:
					j += 1
			
			return test(a,b,i,j,ptol), i, j 
			
		#now do the matching
		match_0, p1_0, p2_0 = lcm_float(np.linalg.norm(p1.Cart[0]),np.linalg.norm(p2.Cart[0]),ptol,n0,m0)
		match_1, p1_1, p2_1 = lcm_float(np.linalg.norm(p1.Cart[1]),np.linalg.norm(p2.Cart[1]),ptol,n1,m1)
		
		if match_0 and match_1:
			"""
			print(p1_0, p1_1, p2_0, p2_1)	
			print(p1_0*np.linalg.norm(p1.Cart[0]), p2_0*np.linalg.norm(p2.Cart[0])) 
			print(p1_1*np.linalg.norm(p1.Cart[1]), p2_1*np.linalg.norm(p2.Cart[1])) 
			"""
			return True, [p1_0, p1_1], [p2_0, p2_1]
		else:
			return False, 0, 0
		

	def tan_match(p1, p2, ptol):
		#compare the tangent of the slab angles, need to be equal within ptol^2 for any hope of a match
		dif = abs(200*(p1.tan_angle-p2.tan_angle)/(p1.tan_angle+p2.tan_angle))
		if dif < ptol**2:
			return True		
		else:
			return False

	#perform a preliminary tan(angle) comparison
	#if this is passed find if the lengths can be matched with multiples, subject to 
	#percentage tolerance and max atom numbers per slab		
	matches = []
	for p1 in pairs1:
		for p2 in pairs2:
			if not tan_match(p1, p2, ptol):
				continue		#written like this to avoid another nested if
		
			#try to match lengths within percetage tolerance
			lmatch, p1_repeats, p2_repeats = length_match(p1, p2, ptol, Rmax)	
			if lmatch:
				#seems to work with deepcopy, with copy the refernce to the list gets copied I think...
				#so that both classes end up refencing the same list and p1 gets modified by changes to P2
				
			
				P1 = Pair(p1.v1, p1.v2, p1.slab, p1.i);			P1.repeat_slab(p1_repeats)
				P2 = Pair(p2.v1, p2.v2, p2.slab, p2.i); 		P2.repeat_slab(p2_repeats)
			
				
				if P1.max_repeats > 0 and P2.max_repeats > 0:
					max_eig = polar_decomp(P1,P2)
					if max_eig < 1 + ptol/100.0:
						matches.append(Match(P1,P2,max_eig))
				

	return matches
			
def angle_filter(pairs):
	"""remove any pairs where the angle is outside the range 60-150 degs"""
	tol = 0.0001
	theta_min = 60.0/180.0 * np.pi - tol
	theta_max = 150.0/180.0 * np.pi + tol
	f_pairs = [x for x in pairs if x.angle > theta_min and x.angle < theta_max]
	return f_pairs

def make_cells(match=None, pair=None):
	#takes an instance of the match class and generates two instances of the cell
	#class that correspond to the bottom and top of the interface
	if match is not None:
		cells = []
		for pair in [match.pair1, match.pair2]:
			slab = pair.slab
			cell = slab.Cell
			origin = slab.origin
			new_Cart = copy.deepcopy(pair.Cart)
			new_cell = copy.deepcopy(cell)
			new_cell.convert_to(new_Cart, origin)
			new_cell.all_frac_from_Cart(wrap=True)
			cells.append(new_cell)	
		
		return cells

	elif pair is not None:
		slab = pair.slab
		cell = slab.Cell
		origin = slab.origin
		new_Cart = copy.deepcopy(pair.Cart)
		new_cell = copy.deepcopy(cell)
		new_cell.convert_to(new_Cart, origin)
		new_cell.all_frac_from_Cart(wrap=True)
			
		return new_cell
				
	
def stack(top_cell, bot_cell, vac=0, Monly=False):
	"""function to return a new instance of the cell class that has the slabs stacked on top of each other"""
	#should write this in terms of energy i.e. input stiffness tensors for each slab then decide on cell to use
	#that minimises the energy		
	
	#for now just add top_cell to bot_cell's system, so take pure bot_cell	
	a1 = top_cell.cell[0];	a2 = top_cell.cell[1];	a3 = top_cell.cell[2]/np.linalg.norm(top_cell.cell[2])
	A = [a1,a2,a3];		A = np.transpose(A)
	b1 = bot_cell.cell[0];	b2 = bot_cell.cell[1];	b3 = bot_cell.cell[2]/np.linalg.norm(bot_cell.cell[2])
	B = [b1,b2,b3];		B = np.transpose(B)
	M = np.dot(B,np.linalg.inv(A))		
	if Monly:
		return M
	
	#change the cartesian vectors of top cell, then update the absolute positions from the fractionals
	top_cell.cell = np.transpose(np.dot(M,np.transpose(top_cell.cell))) #excessive tranposing to keep in terms of column vector
	""" verified that transformation is correct
	print(top_cell.cell, bot_cell.cell)
	"""
	top_cell.all_Cart_from_frac()	

	#wrapping
	#top_cell.all_frac_from_Cart(wrap=True)
	#bot_cell.all_frac_from_Cart(wrap=True)	
	top_cell.all_frac_from_Cart(wrap=False)
	bot_cell.all_frac_from_Cart(wrap=False)	
	
	#calculate s from pair potentials
	s1, s2 = pp_stack_distances(top_cell, bot_cell)
	#print("s1 and s2 are",s1,s2)
	tcell2 = copy.deepcopy(top_cell.cell[2])	
	bcell2 = copy.deepcopy(bot_cell.cell[2])
	#print(bot_cell.cell[2])
	bot_cell.cell[2] += top_cell.cell[2] + s1*bot_cell.cell[2]
	
	#print(bot_cell.cell[2])
	bot_cell.all_frac_from_Cart(wrap=True)
	for i,p in enumerate(top_cell.positions):
		sym = top_cell.symbols[i]
		p2 = p + bot_cell.cell[2]-top_cell.cell[2]
		bot_cell.add_atom(sym, Cart=p2)
	bot_cell.cell[2] += s2 * tcell2	
	bot_cell.all_frac_from_Cart(wrap=True)

	#add vac to cell so vac is extra vaccum over what the pp seps should be
	bot_cell.add_vac(vac/2)	
	
	return bot_cell, M
	


def compare_density(at1, atoms1):
	d1 = len(at1.get_positions())/np.linalg.det(at1.cell)
	d2 = len(atoms1.get_positions())/np.linalg.det(atoms1.cell)

	if abs(d1-d2)*2/(d1+d2) > 10**-5:
		print("unequal number densities")
		print(d1,d2)
		exit()

	return len(at1.get_positions())/len(atoms1.get_positions())

def best_pair(pairs):
	"""function to find the best pair from a list of pairs"""
	#rank first based on atom_count and second on angle
	stats = []
	for i in range(0,len(pairs)):
		stats.append([pairs[i].atom_count, abs(np.sin(pairs[i].angle)), i])

	sorted_pairs = sorted(stats, key=lambda x: (x[0], -x[1]))
	
	return pairs[sorted_pairs[0][2]]

def slab_outfile_name(infile, miller, cr, vac):
	"""standard naming format"""
	outfile = infile + "_"
	m = ""
	for i in miller:
		m +=  str(i)
	outfile += m + "_"
	outfile += str(cr) + "_" + str(vac) + ".res"
	return outfile


def find_ops(atoms):
	"""finds symmetry equivalanet directions of an ase.atoms object"""
	ops = spglib.get_symmetry(atoms)
	new_ops = [(r,t) for r,t in zip(ops["rotations"],ops["translations"])]
	return new_ops

def find_equivalent_directions(atoms, miller1):
	"""finds symmetry equivalent directions of an ser.atoms object"""
	ops = find_ops(atoms)	
	
	miller1 = tuple([x for x in miller1])
	millers = set()
	millers.add(miller1)

	for op in ops:
		R = op[0]
		new = tuple(np.dot(R,miller1))
		millers.add(new)

	return millers
	
def find_equivalent_planes(atoms, miller1):
	"""finds symmetry equivalent planes of an ase.atoms obhect"""
	ops = find_ops(atoms)
	hkl = [[i,miller1[i]] for i in range(0,3)]
	hkl = sorted(hkl, key=lambda x: -x[1])
	print(hkl)

	#find intercepts
	intercepts = []
	for x in hkl:
		if x[1] != 0:
			base = [0,0,0]
			base[x[0]] = 1.0/x[1]
			
		else:
			base = [x for x in intercepts[0]]
			base[x[0]] += 1
		intercepts.append(np.array(base))
	
	#operate on intercepts creating a list of all new intercepts
	all_intercepts = []
	for op in ops:
		new_int = []
		for inter in intercepts:
			new_int.append(np.dot(op[0],inter) + op[1])
		all_intercepts.append(new_int)

	#convert each set of intercepts to miller index representation of plane
	#use sets and tuples to catch repeats
		

	
def pp_stack_distances(top_slab, bot_slab, vac=False):
	"""use pair potentials to find the separation between top and bottom slabs
	to make the forces zero
	if vac is False, default then pairpotentials used to determine distance from top to bottom
	and bottom to top around the loop. If specified then vaccuum is added on top of top so only
	do this once"""

	#TODO strictly should compare to adjacent cells as well but works fine like this 
	#and will be faster. Can imagine some peculiar edge cases that won't work well but 
	#handle those if they come up!!
		
	#read in the covalent radii
	fname = "/u/fs1/jpd47/bit_repos/interface/covalent_radii.dat"
	with open(fname,"r") as f:
		c_rads = {}
		for line in f:
			ws = line.split()
			c_rads[ws[0]] = float(ws[1])		
	
	def E_pp(x, top_slab, bot_slab, c_rads):
		#returns the pairwise potential energy of the slabs
		
		E = 0
		rs = []
		for i,p1 in enumerate(top_slab.positions):
			for j,p2 in enumerate(bot_slab.positions):
				r = p1-p2 + (x[0]+1)*bot_slab.cell[2]
				rs.append(np.linalg.norm(r))
				c_rad1 = c_rads[top_slab.symbols[i]]
				c_rad2 = c_rads[bot_slab.symbols[j]]
				
				E += pair_potential(r, c_rads[top_slab.symbols[i]], c_rads[bot_slab.symbols[j]])
		#print(E,x)
		#print("rmin is", min(rs))
		return E

		
	z0 = [0.2]
	ztop_on_bot = minimize(E_pp, z0, args=( top_slab, bot_slab, c_rads), method='bfgs', options={'disp':False})
	
	zbot_on_top = minimize(E_pp, z0, args=( bot_slab, top_slab, c_rads), method='bfgs', options={ 'disp':False})
	return ztop_on_bot.x, zbot_on_top.x	
		

def pair_potential(cart_sep, r1, r2):
	"""cart_sep is cartesian separation
		r1 and r2 are atomic radii of elements respectively"""
	s = np.linalg.norm(cart_sep)
	
	return (s/(r1+r2))**-12 - (s/(r1+r2))**-6		


def find_best_match(matches):
        """find the best match, based on number of atoms and max_eig"""
        m_sort = sorted(matches, key = lambda x: (x.atom_count, x.max_eig))
        return m_sort[0]	
	


def get_symmetry_translations(cell, parallel_only=True):
	"""parsed an instance of the cell class and return the translational symmetries"""
	
	at = ase.atoms.Atoms(symbols=cell.symbols, positions=cell.positions,cell=cell.cell)
	symmetry = spglib.get_symmetry(at)
	ts = []
	for i,t in enumerate(symmetry["translations"]):
		R = symmetry["rotations"][i]
		if np.allclose(R,np.identity(3)):
			ts.append(t)
	#manually add the full translations
	ts.append([1,0,0])
	ts.append([0,1,0])
	ts.append([0,0,1])
	



	#sort then remove 0 length then remove those not parallel to interface	
	ts_sort = sorted(ts, key=lambda x: (np.linalg.norm(np.dot(cell.cell.T,x))))
	ts_sort = [x for x in ts_sort if np.linalg.norm(np.dot(cell.cell.T,x)) > 1e-05]
	if parallel_only:
		ts_sort = [x for x in ts_sort if np.dot(x, np.cross(cell.cell[0],cell.cell[1])) < 1e-05]	

	#for t in ts_sort:
	#	print(np.linalg.norm(np.dot(cell.cell.T,t)))

	tol = 1e-05
	Ts = [];	Tcarts = []
	Ts.append(ts_sort[0])
	Tcarts.append(np.dot(cell.cell.T,Ts[0]))
	#print("t1, t1_cart", t1, t1_cart)
	for t in ts_sort:
		t_cart = np.dot(cell.cell.T,t)	
		#print("t, t_cart", t, t_cart)
		add = True
		for Tcart in Tcarts:
				if np.linalg.norm(np.cross(Tcart,t_cart)) < tol:
					# parallel
					add = False
				
		if add:
			Ts.append(t)
			Tcarts.append(t_cart)
		if len(Tcarts) > 2:
			break			


	return Tcarts, np.linalg.norm(np.cross(Tcarts[0], Tcarts[1]))	


def write_cell_2_res(interface, args, i,j,n,m):
	"""takes an instance of the Cell class and write an appropriate .res file"""
	
	at3 = ase.atoms.Atoms(symbols=interface.symbols, scaled_positions=interface.frac, cell=interface.cell)	
		
	Area = np.linalg.norm(np.cross(at3.cell[0], at3.cell[1]))
	
	if not args.no_niggli_reduce:
		at3.set_pbc([True, True,True])
		ase.build.niggli_reduce(at3)

	if args.outfile is None:
		m1 = ""
		for x in miller1:
			m1 += str(x)
		if ".cell" in infile1:
			i1 = re.sub(".cell","",infile1)
		elif ".cif" in infile1:
			i1 = re.sub(".cif","",infile1)
				
		if ".cell" in infile2:
			i2 = re.sub(".cell","",infile2)
		elif ".cif" in infile2:
			i2 = re.sub(".cif","",infile2)
		
		m2 = ""
		for x in miller2:
			m2 += str(-x)
		outfile = "" + i1 + "_" + m1 + "_" + str(int(n1)) + "_" +i2 + "_" + m2 + "_" + str(int(n2)) + "_max-eig_" + "{0:0.3f}".format(match.max_eig) + "_Area_" + "{0:0.1f}".format(Area)
		outfile += "_" + str(i)+"of"+str(n)+"_"+str(j)+"of"+str(m)
		outfile += ".res"
	

	ase.io.write(outfile,at3,format="res")
	print("written .res")	

def copy_cell(cell):
	"""copy.deepcopy doesn't work so have to settle for this instead"""
	print("testing copy_cell")
	print("initial volume",np.linalg.det(cell.cell))			
	at = ase.atoms.Atoms(symbols=cell.symbols[:], scaled_positions=cell.frac[:], cell=cell.cell[:])	
	print("final volume", np.linalg.det(at.cell))
	return Cell(at)	


if __name__== '__main__':
	#read in arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("-i1", "--infile1", default="Li_Fm3m.cif")
	parser.add_argument("-o", "--outfile",default=None)
	parser.add_argument("--res", action="store_true")
	parser.add_argument("--no_niggli_reduce", action="store_true", default=False)
	parser.add_argument("-i2", "--infile2", default="Li2S.cif")
	parser.add_argument("-m1","--miller1", default=(1,0,0), nargs="+", type=int)
	parser.add_argument("-m2","--miller2", default=(1,0,0), nargs="+", type=int)
	parser.add_argument("-n1","--max_atoms_1",default=120,type=int,help="max atoms in slab 1")	
	parser.add_argument("-n2","--max_atoms_2",default=120,type=int,help="max atoms in slab 1")	
	parser.add_argument("--slab_only", action="store_true")
	parser.add_argument("-t","--thickness",default=7,type=float)
	parser.add_argument("-cr1","--creps_1",default=None,type=int,help="overrides thickness")
	parser.add_argument("-cr2","--creps_2",default=None,type=int)
	parser.add_argument("-pt","--percentage_tolerance",default=5.0, type=float)
	parser.add_argument("-d","--distance",default=1, type=float)
	parser.add_argument("-v","--vac",default=0, type=float)
	parser.add_argument("-sg","--shift_grid",default=(1,1,), type=float, nargs="+",
			help = "2 integers to specify the size of the shift grid" 	)

	args = parser.parse_args()

	#read the bulk cells and the miller indicies
	infile1 = args.infile1
	miller1 = tuple(args.miller1)
	infile2= args.infile2
	miller2 = tuple(args.miller2)
	res = args.res

	#2 is going on the bottom so m2 ->-m2 to get right orientation of the surface
	miller2 = [-x for x in miller2]


	#split so make all choices for slab1 first. If slab_only specified pick the smallest repeat unit
	#repeat it along c as required, square and output to output file. Then exit
	#idea is this can be used for single slab in vacuum calculations

	#read in atoms using ase and switch to user defined Cell class
	if "cif" in infile1:
		atoms1 = ase.io.read(infile1, format='cif')
	elif "cell" in infile1:
		atoms1 = ase.io.read(infile1, format='castep-cell')

	#cheap hack, if slab only set slab2=slab1
	if args.slab_only:
		atoms2 = copy.deepcopy(atoms1)
	else:
		if "cif" in infile2:
			atoms2 = ase.io.read(infile2, format='cif')
		elif "cell" in infile2:
			atoms2 = ase.io.read(infile2, format='castep-cell')

	"""
	#get symmetry equivalent directions
	symm_dirs = find_equivalent_directions(atoms1, miller1)	
	print(symm_dirs)	
	symm_planes = find_equivalent_planes(atoms1, miller1)
	print(symm_planes)
	"""
	
	cell1 = Cell(atoms1)
	cell2 = Cell(atoms2)

	
	#get the new cell vectors needed to create the desired surface
	slab1 = Slab(miller1, cell1)
	slab1.set_Rmax(args.max_atoms_1)
	slab2 = Slab(miller2,cell2)
	slab2.set_Rmax(args.max_atoms_2)
	#only need to check out as far as the shorter Rmax	
	Rmax = min(slab1.Rmax,slab2.Rmax)	


	#find all vectors in slabs with length <= Rmax
	#returns shortest vector in each direction, none are parallel
	vecs1 = find_vecs(slab1,Rmax)
	vecs2 = find_vecs(slab2,Rmax)	
	
	#construct pairs
	pairs1 = build_pairs(vecs1,slab1)
	pairs2 = build_pairs(vecs2,slab2)


	#TODO insert a symmetry reduction for pairs
	
	#filter pairs, all angles should be between 60 and 150 degrees
	#if a match could be made outside of this range then gauss reduction on the match
	#would bring theta into the desired range and with shorter lattice vectors
	pairs1 = angle_filter(pairs1)
	pairs2 = angle_filter(pairs2)
	matches = match_pairs(pairs1,pairs2,args.percentage_tolerance, Rmax)	



	#if slab only find the best pair in pairs 1 and write out that slab, after repeating ofc
	if args.slab_only:
		pair1 = best_pair(pairs1)			
		out_slab = make_cells(pair=pair1)
	
		
		#convert thicknesses to creps
		if args.creps_1 is not None:
			cr1 = args.creps_1
		else:
			c = np.linalg.norm(out_slab.cell[2])	
			cr1 = int(np.ceil(args.thickness/c))

		#repeat the slab, replace with function
		out_slab.crep(cr1)	
		if args.vac > 0.001:			
			out_slab.square()
			out_slab.add_vac(0.5* args.vac) 
			
		#write out the slab 
		at1 = ase.atoms.Atoms(symbols=out_slab.symbols, scaled_positions=out_slab.frac, cell=out_slab.cell)
		if args.outfile is None:
			outfile = slab_outfile_name(infile1, miller1, cr1, args.vac)
		ase.io.write(outfile,at1,format="res")
		exit()


	#TODO symmetry reduce matches, do it here or might be easier/more efficient to do
	#it earlier, e.g. symmetry reduce the vectors before pairs are constructed
	#that won't get all symetry equivalents though... maybe it is best just to reduce
	#matches...??
	match = find_best_match(matches)

	#match = matches[0]
	print("there are ",len(matches), "matches")
	
	#pair 1 is bottom cell, pair 2 is top cell
	top_cell, bot_cell = make_cells(match=match)
	print("made cells")


	"""	
	#testing niggli reduction	
	test_cell = ase.atoms.Atoms(symbols=top_cell.symbols, positions=top_cell.positions,cell=top_cell.cell)
	
	#shows that original cell vectors are now translational symmetries of top_cell
	t1,t2, Area = get_symmetry_translations(top_cell)	
	print(np.linalg.norm(t1))	
	print(np.linalg.norm(t2))	
	for c in atoms1.cell:
		print(np.linalg.norm(c))	
	
	print(np.linalg.det(test_cell.cell))
	test_cell.set_pbc([True, True, True])
	ase.build.niggli_reduce(test_cell)
	print(np.linalg.det(test_cell.cell))
	
	print(np.linalg.det(atoms1.cell))
	exit()	

	"""
	#convert thicknesses to creps
	if args.creps_1 is not None:
		cr1 = args.creps_1
	else:
		c = np.linalg.norm(top_cell.cell[2])	
		cr1 = int(np.ceil(args.thickness/c))


	if args.creps_2 is not None:
		cr2 = args.creps_2
	else:
		c = np.linalg.norm(bot_cell.cell[2])	
		cr2 = int(np.ceil(args.thickness/c))

	top_cell.crep(cr1)
	bot_cell.crep(cr2)

	

	print("c repeated cells")

	#square the slabs and then stack themn		
	bot_cell.square()
	top_usq  = copy_cell(top_cell)
	top_cell.square()


	"""
	#shows that original cell vectors are now translational symmetries of top_cell
	ts, Area = get_symmetry_translations(top_usq, parallel_only=False)
	print("ts")
	for t in ts:
		print(np.linalg.norm(t))
	print("cells")
	for c in atoms1.cell:
		print(np.linalg.norm(c))	

	print(np.linalg.det(np.array(ts)))
	print(np.linalg.det(atoms1.cell))	
	print(np.linalg.det(top_cell.cell))
	"""


	#get the transormation matrix that used to strain the top_cell
	#apply it, reduce to primitive and then write out as bulk.res
	
	print(top_usq.cell, np.linalg.det(top_usq.cell))
	M = stack(top_cell, bot_cell, Monly=True)
	
	print(top_usq.cell, np.linalg.det(top_usq.cell))
#	M = 5*np.identity(3)
	
	print("M is", M)
	top_usq.cell = np.transpose(np.dot(M,np.transpose(top_usq.cell))) 
	
	print(top_usq.cell, np.linalg.det(top_usq.cell))
	#top_usq.all_Cart_from_frac()
	
	print(top_usq.cell, np.linalg.det(top_usq.cell))
	#print(top_usq.__dict__)
	ats = ase.atoms.Atoms(symbols=top_usq.symbols, scaled_positions=top_usq.frac,cell=top_usq.cell)
	print(ats.get_scaled_positions())
	print(top_usq.cell, np.linalg.det(top_usq.cell))
	lattice, scaled_positions, numbers = spglib.find_primitive(ats)
	
	print(top_usq.cell, np.linalg.det(top_usq.cell))
	print(lattice) 
	print(np.linalg.det(lattice), np.linalg.det(top_usq.cell))

	sys.exit(0)	
	"""
	#use stack to get transformation matrix for top_cell
	#apply, reduce to primitive and write out to be used to find
	#the strain energy
	
	M = stack(top_cell, bot_cell, Monly=True)
	top_usq.cell = np.transpose(np.dot(M,np.transpose(top_usq.cell))) 
	top_usq.all_frac_from_Cart(wrap=True)	
	strained = ase.atoms.Atoms(symbols=top_usq.symbols, positions=top_usq.positions,cell=top_usq.cell)
	print(np.linalg.det(strained.cell))
	strained.set_pbc([True, True, True])
	ase.build.niggli_reduce(strained)
	print(np.linalg.det(strained.cell))
	
	print(np.linalg.det(atoms1.cell))
	exit()	
	"""
	
	#write the top and bottom cell out as .cell files
	at1 = ase.atoms.Atoms(symbols=top_cell.symbols, positions=top_cell.positions,cell=top_cell.cell)
	at1.set_pbc([True,True, False])
	
	#ase.io.write("top.cell",at1,format="castep-cell")
	at2 = ase.atoms.Atoms(symbols=bot_cell.symbols, scaled_positions=bot_cell.frac,cell=bot_cell.cell)
	#ase.io.write("bot.cell",at2,format="castep-cell")
	at2.set_pbc([True, True, False])
	symmetry = spglib.get_symmetry(at1)	

	
	#check number densities of the slabs
	n1 = compare_density(at1, atoms1)
	n2 = compare_density(at2, atoms2)


	#get the symmetry translation vectors for each slab
	tt1, tt2, tA = get_symmetry_translations(top_cell)
	bt1, bt2, bA = get_symmetry_translations(bot_cell)	
	#as shifts are relative, take the shifts with smaller area
	#BUT can always shift the top cell w.l.o.g
	if tA < bA:
		shifts = [tt1,tt2]
	else:
		shifts = [bt1,bt2]

	#	shift the final section into a function called stack_and_write
	#then create list of shifted cells to pass to stack_and_write 
	#so will end up with one call generating all the required shifts	
	n,m = args.shift_grid
	n = int(n);		m = int(m)
	for i in range(0,n):
		for j in range(0,m):
			t1 = shifts[0]/n * i
			t2 = shifts[1]/m * j

	
			#apply the shifts and then create the interface
			bot_cell0 = copy_cell(bot_cell)		
			top_cell0 = copy_cell(top_cell)
			bot_cell0.shift(t1+t2)
			interface = stack(top_cell0, bot_cell0, vac=args.vac)		
			write_cell_2_res(interface, args, i,j,n,m)


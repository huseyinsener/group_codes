#! /usr/bin/env python

import os, sys
import subprocess as sp
import argparse
vasppp = os.environ['VASP_PP_PATH']

def get_potcar(elements, xc):
    if xc.lower() == 'pbe':
        ppdir = os.path.join(vasppp, 'pot_PAW_PBE')
    elif xc.lower() == 'pbe_new':
        ppdir = os.path.join(vasppp, 'pot_PAW_PBENEW')
    elif xc.lower() == 'lda_new':
        ppdir = os.path.join(vasppp, 'pot_PAW_LDANEW')

    pp = ''
    for e in elements:
        print os.path.join(ppdir, e, 'POTCAR')
        pp += open(os.path.join(ppdir, e, 'POTCAR')).read()

    return pp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--elements', nargs='*')
    parser.add_argument('-f', '--poscar', default='POSCAR')
    parser.add_argument('-xc', '--xc', choices=['pbe', 'pbe_new', 'lda_new'], default='pbe')
    parser.add_argument('-o', '--out_potcar', default='POTCAR')

    args = parser.parse_args()

    if args.elements:
        elements = args.elements
    else:
        try:
            elements = open(args.poscar).readlines()[0].strip().split()
        except:
            print 'Elements not given on command line and POSCAR not found'
            sys.exit(-1)

    pp = get_potcar(elements, args.xc)

    p = open(args.out_potcar, 'w')
    p.write(pp)
    p.close()

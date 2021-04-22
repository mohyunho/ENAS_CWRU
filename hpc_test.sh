#!/bin/bash 
#PBS -l select=1:mem=128gb   
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -m abe
#PBS -M hyunho.mo@unitn.it
#PBS -o results_hp1.out
#PBS -q common_cpuQ
source ~/home/hyunhomo/sn_p/bin/activate
cd ~/home/hyunhomo/ENAS_CWRU/
python3 enas_cwru.py -i 48 --hp 1 --pop 30 --gen 30


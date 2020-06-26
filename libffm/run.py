#!/usr/bin/env python3

import subprocess, sys, os, time

NR_THREAD = 8

start = time.time()

cmd = './utils/count.py tr.csv > fc.trva.t10.txt'
print(cmd)
subprocess.call(cmd, shell=True) 

cmd = 'converters/parallelizer-a.py -s {nr_thread} converters/pre-a.py tr.csv tr.gbdt.dense tr.gbdt.sparse'.format(nr_thread=NR_THREAD)
print(cmd)
subprocess.call(cmd, shell=True) 

cmd = 'converters/parallelizer-a.py -s {nr_thread} converters/pre-a.py te.csv te.gbdt.dense te.gbdt.sparse'.format(nr_thread=NR_THREAD)
print(cmd)
subprocess.call(cmd, shell=True) 

cmd = './gbdt -t 30 -s {nr_thread} te.gbdt.dense te.gbdt.sparse tr.gbdt.dense tr.gbdt.sparse te.gbdt.out tr.gbdt.out'.format(nr_thread=NR_THREAD) 
print(cmd)
subprocess.call(cmd, shell=True)


cmd = 'converters/parallelizer-b.py -s {nr_thread} converters/pre-b.py tr.csv tr.gbdt.out tr.ffm'.format(nr_thread=NR_THREAD)
print(cmd)
subprocess.call(cmd, shell=True) 

cmd = 'converters/parallelizer-b.py -s {nr_thread} converters/pre-b.py te.csv te.gbdt.out te.ffm'.format(nr_thread=NR_THREAD)
print(cmd)
subprocess.call(cmd, shell=True) 


cmd = './ffm-train --on-disk --no-rand -k 5 -t 25 -l 0.000003 -s {nr_thread} tr.ffm mode'.format(nr_thread=NR_THREAD) 
print(cmd)
subprocess.call(cmd, shell=True)

cmd = './ffm-predict te.ffm model te.out'.format(nr_thread=NR_THREAD) 
print(cmd)
subprocess.call(cmd, shell=True)



cmd = './utils/make_submission.py te.out submission_ffm.csv'
print(cmd)
subprocess.call(cmd, shell=True)

print('time used = {0:.0f}'.format(time.time()-start))

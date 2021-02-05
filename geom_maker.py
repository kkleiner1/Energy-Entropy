import os
import glob
import numpy as np

system_length_array = np.array([['h2',1],['h2',1.4],['h2',2],['h2',3],['h2',4]])

def make_paths(system_length_array):
    lengths = system_length_array[:,1]
    systems= system_length_array[:,0]
    for sys in systems:
        for l in lengths:
            path = f'sys_{sys}_len_{l}'
            try:
                os.mkdir(path)
            except OSError as error:  
                print(error)   

make_paths(system_length_array)

def make_geometries():
    for x in glob.glob('sys_*_len*'):
        print(x)
        length = x.split('_')[-1]
        print(length)
        string = f'H 0 0 0; H 0 0 {length}'
        f = open(f"{x}/geom.xyz","w")
        f.write(string)
        f.close
make_geometries()


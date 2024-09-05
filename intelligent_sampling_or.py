from pathlib import Path
import sys
import os
import numpy as np
from xanesnet.descriptor.wacsf import WACSF
from xanesnet.utils import load_xyz
from sklearn.decomposition import PCA
import shutil
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser


def parse_args(args):

    p = ArgumentParser()

    help_strs = {'xyz': 'path to xyz dir', 'spectra': 'path to spectra dir', 'min': 'starting value for IT',
            'max': 'maximum value for IT', 'step': 'step size', 'xyz_xanes':'choose to sample by xanes or xyz'}

    p.add_argument('xyz_dir',type=str,help=help_strs['xyz'])
    p.add_argument('xanes_dir', type=str, help=help_strs['spectra'])
    p.add_argument('minimum', type=int, help=help_strs['min'])
    p.add_argument('maximum', type=int, help=help_strs['max'])
    p.add_argument('step_size', type=int, help=help_strs['step'])
    p.add_argument('choice', type=str, help=help_strs['xyz_xanes'])

    args = p.parse_args()

    return args


def intelligent_sample(xyz, xanes, min, max, step):

    n_sample_list = np.arange(min,(max+1),step).tolist()

    wacsf_ob =  WACSF(n_g2 = 32, n_g4 = 64, r_min = 1, r_max = 6, z = [1, 2, 4, 8, 16, 32, 64, 128])
    xyz_f_list = [f for f in xyz.iterdir()]
    f_list = [f.stem for f in xyz_f_list]
    ids = [str(f) + '.xyz' for f in f_list]
    print('Loading descriptors, this may take a minute...')
    descs = []
    i = 0
    for f in xyz_f_list:
        try:
            with open(f, 'r') as g:
               desc = wacsf_ob.transform(load_xyz(g))
               descs.append(desc)
        except Exception as e:
            print(f'Error loading descriptor at index {i}: {e}')
            print(i,f)
        i = i + 1
        print(f'Loaded descriptors for {i} files out of {len(xyz_f_list)}')
        
    print('Loaded descriptors, performing intelligent sampling...')

    for n in n_sample_list:
        print(f'Starting intelligent sampling for n = {n}')
        s1 = StandardScaler()
        x1 = s1.fit_transform(descs)
        pca = PCA(n_components=3)
        points = pca.fit_transform(x1)

        n_samples = n
        selected_xy = np.zeros([n_samples, 3])
        ids_to_keep = []
        points_left = np.arange(len(points))
        sample_inds = np.zeros(n_samples, dtype='int')
        ids_to_keep = np.empty(n_samples, dtype='object')
        dists = np.ones_like(points_left) * float('inf')
        selected = 0
        sample_inds[0] = points_left[selected]
        points_left = np.delete(points_left, selected)

        for i in range(1, n_samples):
            last_added = sample_inds[i - 1]

            dist_to_last_added_point = (
                    (points[last_added] - points[points_left]) ** 2).sum(-1)  #

            dists[points_left] = np.maximum(dist_to_last_added_point,
                                            dists[points_left])

            selected = np.argmax(dists[points_left])
            sample_inds[i] = points_left[selected]
            points_left = np.delete(points_left, selected)
            selected_xy[i, :] = points[sample_inds[i]]
            ids_to_keep[i] = ids[sample_inds[i]]

        spectra_to_keep = []
        for f in ids_to_keep[1:]:
            f = os.path.splitext(f)[0]
            f = f + '.txt'
            spectra_to_keep.append(f)


        Path(f'./IT_data/{n}/xyz').mkdir(parents=True, exist_ok=True)
        Path(f'./IT_data/{n}/spectra').mkdir(parents=True, exist_ok=True)
        
        print("selected_xy:", selected_xy)
        print("ids_to_keep:", ids_to_keep)
        print("spectra_to_keep:", spectra_to_keep)
    
    
    
        for file in xyz.iterdir():

            if file.stem + '.xyz' in ids_to_keep:
                shutil.copy(file, f'./IT_data/{n}/xyz')

        for file in xanes.iterdir():

            if file.stem + '.txt' in spectra_to_keep:
                shutil.copy(file, f'./IT_data/{n}/spectra')

        print(f'Done {n}')


def it_spectra(xyz, xanes, min, max, step):
    n_sample_list = np.arange(min,max,step).tolist()

    xanes_f_list = [f for f in xanes.iterdir()]
    f_list = [f.stem for f in xanes_f_list]
    ids = [str(f) + '.txt' for f in f_list]
    spectra = []
    for f in xanes.iterdir():
        with open(f, 'r') as g:
            spec = np.genfromtxt(g, skip_header=2, usecols=1)
        spectra.append(spec)

    for n in n_sample_list:
        print(f'Starting intelligent sampling for n = {n}')
        pca = PCA(n_components=3)
        points = pca.fit_transform(spectra)
        n_samples = n
        selected_xy = np.zeros([n_samples, 3])
        ids_to_keep = []
        points_left = np.arange(len(points))
        sample_inds = np.zeros(n_samples, dtype='int')
        ids_to_keep = np.empty(n_samples, dtype='object')
        dists = np.ones_like(points_left) * float('inf')
        selected = 0
        sample_inds[0] = points_left[selected]
        points_left = np.delete(points_left, selected)

        for i in range(1, n_samples):
            # Find the distance to the last added point in selected
            # and all the others
            last_added = sample_inds[i-1]

            dist_to_last_added_point = (
                (points[last_added] - points[points_left])**2).sum(-1) # [P - i]

            # If closer, updated distances
            dists[points_left] = np.minimum(dist_to_last_added_point,
                                            dists[points_left]) # [P - i]

            # We want to pick the one that has the largest nearest neighbour
            # distance to the sampled points
            selected = np.argmax(dists[points_left])
            sample_inds[i] = points_left[selected]
            points_left = np.delete(points_left, selected)
            selected_xy[i,:] = points[sample_inds[i]]
            ids_to_keep[i] = ids[sample_inds[i]]

        spectra_to_keep = []
        for f in ids_to_keep[1:]:
            f = os.path.splitext(f)[0]
            f = f + '.xyz'
            spectra_to_keep.append(f)

        Path(f'./IT_data/{n}/xyz').mkdir(parents=True, exist_ok=True)
        Path(f'./IT_data/{n}/spectra').mkdir(parents=True, exist_ok=True)


        for file in xanes.iterdir():

            if file.stem + '.txt' in ids_to_keep:
                shutil.copy(file, f'./IT_data/{n}/spectra')

        for file in xyz.iterdir():

            if file.stem + '.xyz' in spectra_to_keep:
                shutil.copy(file, f'./IT_data/{n}/xyz')

        print(f'Done {n}')



def main(args):

    args = parse_args(args)
    print(f"Arguments: {args}")
    if args.choice == 'xyz':
        print("Running intelligent_sample function for XYZ...")
        intelligent_sample(Path(args.xyz_dir), Path(args.xanes_dir), args.minimum, args.maximum, args.step_size)
    if args.choice == 'xanes':
        print("Running it_spectra function for XANES...")
        it_spectra(Path(args.xyz_dir), Path(args.xanes_dir), args.minimum, args.maximum, args.step_size)


if __name__ == '__main__':
    main(sys.argv[1:])

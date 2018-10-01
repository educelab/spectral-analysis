import h5py

hdf5_file = '/Users/stephen/temp/spectral/2017_07_18_15_51_56/analysis/some_vectors.hdf5'
list_file = '/Users/stephen/repos/spectral-analysis/ordered_images.txt'
sorted_file = '/Users/stephen/temp/spectral/2017_07_18_15_51_56/analysis/sorted.hdf5'

with open(list_file, 'r') as f:
    names = [line for line in f]

ids = [j for (name, j) in sorted([(name, i) for (i, name) in enumerate(names)])]
first = ids[0]
del(ids[0])
ids.append(first)

with h5py.File(hdf5_file, 'r') as f:
    original_dset = f['points']

    with h5py.File(sorted_file, 'w') as f:
        dset = f.create_dataset('points', (100000, 370), dtype='f')    

        for p in range(100000):
            dset[p] = [data for (j, data) in sorted(zip(ids, original_dset[p]))]
            # print(original_dset[p])
            # print(dset[p])
            # print()
            # g = input()

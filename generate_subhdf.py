import numpy as np
import pandas as pd
import h5py
import tqdm

if __name__ == '__main__':
    import argparse
    from warnings import warn

    parser = argparse.ArgumentParser(description='get partial hdf5 with subset from the entries of an original '\
                        + 'hdf5 file.')
    parser.add_argument('path_to_input_hdf', type=str, default='ecg_tracings.hdf5',
                        help='path to hdf5 containing ECG traces.')
    parser.add_argument('path_to_csv', type=str, default='exams_info.csv',
                        help='path to csv file containing exams info with classes.')
    #parser.add_argument('path_to_ids', type=str,
                        #help='path to txt containing ids.')
    parser.add_argument('--traces_dset', default='signal',
                        help='traces dataset in the hdf5 file.')
    parser.add_argument('path_to_output_hdf', type=str,
                        help='path to new hdf5 containing ECG traces.')
    parser.add_argument('path_to_output_csv', type=str,
                        help='path to new csv file containing new exams info.')
    parser.add_argument('--ids_dset', default='id_exam',
                        help='by default consider the ids are just the order')
    parser.add_argument('--ids_col', default='id_exam',
                        help='column with the ids in csv file.')
    parser.add_argument('--class_col', default='exam_class',
                        help='column with the exams classes in csv file.')
    parser.add_argument('-b', '--batch_size', type=int, default=128,
                        help='number of exams per batch.')
    parser.add_argument('-m', '--maximum_ids', type=int, default=-1,
                        help='maximum ids to consider.')
    parser.add_argument('-n', '--minimum_ids', type=int, default=-1,
                        help='minimum ids to consider. The starting point until max or end')

    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")
    print(args)
    # load ids
    #   new_ids = np.loadtxt(args.path_to_ids, dtype=np.int)
    
    # load dataframe with exams info
    df = pd.read_csv(args.path_to_csv, index_col=args.ids_col)
    
    # Get h5 data and Load hdf dataset
    f = h5py.File(args.path_to_input_hdf, 'r')
    signal = f[args.traces_dset]
    # get dimension
    _, seq_len, n_leads = f['signal'].shape
    
    # keep ids that are present in hdf5 file, in df
    if args.ids_dset:
        h5ids = f[args.ids_dset]
        df = df.reindex(h5ids, fill_value=False, copy=True)
    
    new_ids = list(df.loc[(df[args.class_col] != 0)].index)  # ids for which the exams are in class 1, 2 and 3
    
    
    # Get ids
    ids = np.array(f[args.ids_dset])
    #final_ids = np.intersect1d(ids, new_ids) # return the sorted, unique values that are in both of the input arrays
    #idx = np.isin(ids, final_ids)
    idx = np.isin(ids, new_ids) # Returns a boolean array of the same shape as ids that is True for values in both arrays
    
    # Verify the length of new ids and sum(idx)
    print("Number of new ids are equal: ", (len(new_ids) == sum(idx)))

    # Verify that there is no NaN under the entire DataFrame
    print("There is no NaN values in new dataframe: ", ~df.isnull().values.any())

    
    if args.maximum_ids > 0:
        idx = idx & (np.arange(len(idx)) < args.maximum_ids)
    if args.minimum_ids > 0:
        idx = idx & (np.arange(len(idx)) >= args.minimum_ids)
    n_total = len(ids)
    counter = 0
    
    # Generate hdf5
    incr_aux = np.arange(len(idx))
    out_f = h5py.File(args.path_to_output_hdf, "w")
    ids_dset_new = out_f.create_dataset("id_exam", dtype=np.int64, shape=(sum(idx),))  # Save ids
    signal_new = out_f.create_dataset("signal", dtype=np.float32, shape=(sum(idx), seq_len, n_leads))
    end = 0
    n_batches = int(np.ceil(n_total/args.batch_size))
    counter = 0
    for i in tqdm.tqdm(range(n_batches)):
        start = end
        end = min((i + 1) * args.batch_size, n_total)
        mask = idx[start:end]
        sum_mask = sum(mask)
        if sum_mask > 0:
            ids_batch = ids[start:end]
            signal_batch = signal[start:end, :, :]
            ids_dset_new[counter:counter+sum_mask] = ids_batch[mask]
            signal_new[counter:counter+sum_mask, :, :] = signal_batch[mask, :, :]
        counter += sum_mask
    print("No of traces in a new dataset: ", counter)
    # Create dataset
    out_f.close()
    
    # Saving information of the selected exam traces in a csv file
    df1 = df.reset_index()
    df1 = df1[idx]
    # Confirm that the number of items are equal
    print("Number of items in hdf5 and csv files are equals: ", len(df1) == counter)
    print("\nOriginal exams info file (exams with classes): \n", df1.head(10))
    print("Number of items in csv file: ", len(df1))
    # Save a new csv file
    new_df = pd.DataFrame(df1[['index', 'id_patient' , 'age', 'sex', 'AF', 'exam_class', 'split']])
    new_df.rename(columns={"index": "id_exam"}, inplace=True)
    new_df.to_csv(args.path_to_output_csv)

    # Verification if similarity in ids
    f2 = h5py.File(args.path_to_output_hdf, 'r')
    newh5ids = np.array(f2[args.ids_dset])
    new_df2 = pd.read_csv(args.path_to_output_csv)
    new_df2ids = np.array(new_df2['id_exam'])
    print("Ids in new h5 and new csv are equal: ", np.unique(newh5ids == new_df2ids))
    print("\nnew exams info csv file: \n", new_df2.head(10))

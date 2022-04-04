# Imports
from resnet import ResNet1d
import tqdm
import h5py
import torch
import os
import json
import numpy as np
import argparse
from warnings import warn
import pandas as pd
import torch.nn.functional as F
from dataloader import BatchDataloader


def get_accuracy(Y_hat, Y):
    #torch.argmax(input) → LongTensor :-> Returns the indices of the maximum value of all
    #elements in the input tensor. (dim = 1 -> per row)
    #return (Y_hat == Y).float().mean()
    return (Y_hat == Y).mean()



if __name__ == "__main__":
    parser = argparse.ArgumentParser( description='Evaluate AF class prediction model.', add_help=False)
    parser.add_argument('mdl', type=str,
                        help='folder containing model.')
    parser.add_argument('path_to_traces', type=str, default='../data/ecg_tracings.hdf5',
                        help='path to hdf5 containing ECG traces')
    parser.add_argument('--path_to_csv', type=str, default='',
                        help='path to csv file containing exams info with classes for test data.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='number of exams per batch.')
    parser.add_argument('--output', type=str, default='predicted_af.csv',
                        help='output file.')
    parser.add_argument('--traces_dset', default='signal',
                         help='traces dataset in the hdf5 file.')
    parser.add_argument('--ids_dset', default='id_exam',
                         help='ids dataset in the hdf5 file.')
    parser.add_argument('--ids_col', default='id_exam',
                        help='column with the ids in csv file.')
    parser.add_argument('--class_col', default='exam_class',
                        help='column with the exams classes in csv file.')
    parser.add_argument('--split_col', default='split',
                        help='column with train-test split info in csv file.')
    parser.add_argument('--full_dset', default= True,
                        help='whether the input dataset contains training and test samples')
    parser.add_argument('--n_valid', type=int, default=-1,
                        help='the first `n_valid` exams in the hdf will be considered during testing,'
                             'when full_dset is True. Only exams labeled as test will be used.')

    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")
    print(args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Get checkpoint
    ckpt = torch.load(os.path.join(args.mdl, 'model.pth'), map_location=lambda storage, loc: storage)
    # Get config
    config = os.path.join(args.mdl, 'args.json')
    with open(config, 'r') as f:
        config_dict = json.load(f)
    # Get model
    N_LEADS = 12
    N_CLASSES = 3  # Classes 1, 2 and 3
    model = ResNet1d(input_dim=(N_LEADS, config_dict['seq_length']),
                     blocks_dim=list(zip(config_dict['net_filter_size'], config_dict['net_seq_lengh'])),
                     n_classes=N_CLASSES,  # Classes 1, 2 and 3
                     kernel_size=config_dict['kernel_size'],
                     dropout_rate=config_dict['dropout_rate'])
    # load model checkpoint
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    # Get traces
    ff = h5py.File(args.path_to_traces, 'r')
    traces = ff[args.traces_dset]
    n_total = len(traces)
    if args.ids_dset:
        ids = ff[args.ids_dset]
    else:
        ids = range(n_total)

    #predicted_class = np.zeros((n_total,))

    # Evaluate on test data
    model.eval()

    if args.full_dset:
        exam_info_df = pd.read_csv(args.path_to_csv, index_col=args.ids_col)
        exam_info_df = exam_info_df.reindex(ids, fill_value=False, copy=True)

        af_classes = exam_info_df[args.class_col]

        '''
        For testing the model on full dataset "traces100pc" , can select few traces with n_valid
        Can also use traces15pc without class 0 (pre-selected).
        '''

        if args.n_valid > 0:
            mask_chunk = np.arange(len(exam_info_df)) < args.n_valid
        else:
            mask_chunk = np.arange(len(exam_info_df)) < len(exam_info_df)  # Full dataset

        test_mask_0 = (exam_info_df[args.split_col] == "test").to_numpy()
        test_mask = test_mask_0 & mask_chunk
        n_total = sum(test_mask) #total number of samples used for testing

        test_loader = BatchDataloader(traces, af_classes, bs=args.batch_size, mask=test_mask)

        predicted_class = np.zeros((n_total,N_CLASSES + 1))  # For class probabilities and predicted class
        end = 0

        for traces, af_classes in tqdm.tqdm(test_loader):
            start = end
            end += len(traces)

            with torch.no_grad():
                traces = traces.transpose(1, 2)	# traces refer to x
                traces = traces.to(device)
                y_pred = model(traces)
                y_pred = F.softmax(y_pred, dim=1)
                y_prob = y_pred
                y_pred = y_pred.argmax(dim=1) + 1
            predicted_class[start:end,:N_CLASSES] = y_prob.detach().cpu().numpy()
            predicted_class[start:end, -1] = y_pred.detach().cpu().numpy()

        # Add normalized probabilities of classes 3 and 1
        prob_sum = predicted_class[:,0] + predicted_class[:,2]
        prob_class3_norm = predicted_class[:,2] / prob_sum
        prob_class1_norm = predicted_class[:,0] / prob_sum

        # Save predictions
        test_ids = ids[test_mask]
        df = pd.DataFrame({'exam_ids': test_ids, 'predicted_class': predicted_class[:,-1],
                           'prob_class1' : predicted_class[:,0],
                           'prob_class2' : predicted_class[:,1],
                           'prob_class3' : predicted_class[:,2],
                           'prob_class1_norm' : prob_class1_norm,
                           'prob_class3_norm' : prob_class3_norm })
        df = df.set_index('exam_ids', drop=False)
        df.to_csv(args.output)

        # Calculate accuracy
        if args.path_to_csv:
            df2 = pd.read_csv(args.path_to_csv, index_col=args.ids_col)
            df2 = df2.reindex(test_ids, fill_value=False, copy=True) # Keep ids that are in h5 file
            y_true = np.array(df2[args.class_col])
            accuracy = get_accuracy(predicted_class[:,-1], y_true)
            print(f'The accuracy is: {accuracy * 100:.3f}%.')
            #print('The accuracy is: {:6.3f}%.'.format(accuracy * 100))

    else:
        # Get dimension
        n_total, n_samples, n_leads = traces.shape
        n_batches = int(np.ceil(n_total/args.batch_size))
        # Compute gradients
        predicted_class = np.zeros((n_total,N_CLASSES + 1))  # For class probabilities and predicted class
        end = 0
        for i in tqdm.tqdm(range(n_batches)):
            start = end
            end = min((i + 1) * args.batch_size, n_total)
            with torch.no_grad():
                x = torch.tensor(traces[start:end, :, :]).transpose(-1, -2)
                x = x.to(device, dtype=torch.float32)
                y_pred = model(x)
                y_pred = F.softmax(y_pred, dim=1)
                y_prob = y_pred
                y_pred = y_pred.argmax(dim=1) + 1
            predicted_class[start:end,:N_CLASSES] = y_prob.detach().cpu().numpy()
            predicted_class[start:end, -1] = y_pred.detach().cpu().numpy()        #.astype(int)

        # Add normalized probabilities of classes 3 and 1
        prob_sum = predicted_class[:,0] + predicted_class[:,2]
        prob_class3_norm = predicted_class[:,2] / prob_sum
        prob_class1_norm = predicted_class[:,0] / prob_sum

        # Save predictions
        df = pd.DataFrame({'exam_ids': ids, 'predicted_class': predicted_class[:,-1],
                           'prob_class1' : predicted_class[:,0],
                           'prob_class2' : predicted_class[:,1],
                           'prob_class3' : predicted_class[:,2],
                           'prob_class1_norm' : prob_class1_norm,
                           'prob_class3_norm' : prob_class3_norm })
        df = df.set_index('exam_ids', drop=False)
        df.to_csv(args.output)

        # Calculate accuracy
        if args.path_to_csv:
            df2 = pd.read_csv(args.path_to_csv, index_col=args.ids_col)
            h5ids = ff[args.ids_dset]
            df2 = df2.reindex(h5ids, fill_value=False, copy=True) # Keep ids that are in h5 file
            y_true = np.array(df2[args.class_col])
            accuracy = get_accuracy(predicted_class[:,-1], y_true)
            print(f'The accuracy is: {accuracy * 100:.3f}%.')
            #print('The accuracy is: {:6.3f}%.'.format(accuracy * 100))

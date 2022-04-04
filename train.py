import json
import torch
import os
from tqdm import tqdm
from resnet import ResNet1d
import torch.nn.functional as F
from dataloader import BatchDataloader
import torch.optim as optim
import numpy as np

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize


def compute_loss(ages, pred_ages, weights):
    diff = torch.abs(ages.flatten() - pred_ages.flatten())
    loss = torch.sum(weights.flatten() * diff * diff)
    diff_w = torch.sum(weights.flatten() * diff)
    return loss

def crossentropy(G, Y):
    '''Compute loss, classification case'''
    # convert labels to onehot encoding
    n_classes = 3
    G = F.softmax(G, dim=1)
    Y_onehot = torch.eye(n_classes, device=device)[Y-1]

    #return -(Y_onehot * G.log()).sum(dim = 1).sum(dim=0)
    return -(Y_onehot * G.log()).sum()

def compute_weights(classes):
    _, inverse, counts = np.unique(classes, return_inverse=True, return_counts=True)
    weights = 1 / counts
    normalized_weights = weights / sum(weights)
    w = normalized_weights / min(normalized_weights)
    #w = w * np.array([1, 5, 200])

    return w

def compute_metrics(all_logits, all_labels):
    """Compute ROC_AUC and Average Precision after evaluation of trained model, per epoch."""
    metrics = []
    fpr = {}
    tpr = {}
    thresh = {}
    auc_scores = []
    #ap_scores = []
    n_classes = all_logits.size(1)
    true_onehot = label_binarize(all_labels, classes=[1, 2, 3])
    all_logits = F.softmax(all_logits, dim=1)
    average_precision = []

    # Compute AUC scores (one vs rest)
    for i in range(n_classes):
        fpr[i], tpr[i], thresh[i] = roc_curve(all_labels, all_logits[:,i], pos_label=i+1)
        auc_scores.append(auc(fpr[i], tpr[i]))

    # Compute Average Precision Scores
    for i in range(n_classes):
        average_precision.append(average_precision_score(true_onehot[:, i], all_logits[:, i], average='micro'))

    # Removing samples belonging to class 2
    three_one_mask = all_labels != 2
    three_one_logits = all_logits[three_one_mask]
    three_one_onehot = true_onehot[three_one_mask]

    # Compute the AUC score of class 3 VS class 1
    prob_sum = three_one_logits[:, 2] + three_one_logits[:, 0]
    prob_class3_norm = three_one_logits[:, 2] / prob_sum
    #prob_class1_norm = three_one_logits[:, 0] / prob_sum
    three_vs_one_auc_score = roc_auc_score(three_one_onehot[:, 2], prob_class3_norm, average=None)

    # Compute Average Precision Score of class 3 VS class 1
    three_vs_one_ap_score = average_precision_score(three_one_onehot[:, 2], prob_class3_norm, average='micro')

    metrics.append(auc_scores)
    metrics.append(average_precision)
    metrics.append([three_vs_one_auc_score, three_vs_one_ap_score])

    return metrics


def train(ep, dataload, weights=None):
    model.train()
    total_loss = 0
    n_entries = 0
    train_desc = "Epoch {:2d}: train - Loss: {:.6f}"
    train_bar = tqdm(initial=0, leave=True, total=len(dataload),
                     desc=train_desc.format(ep, 0, 0), position=0)
    for traces, af_classes in dataload:
        traces = traces.transpose(1, 2)
        traces, af_classes = traces.to(device), af_classes.to(device)
        # Reinitialize grad
        model.zero_grad()
        # Send to device
        # Forward pass
        pred_classes = model(traces)
        #loss = compute_loss(ages, pred_ages, weights)
        pred_classes = pred_classes.type(torch.DoubleTensor)  # float type raises error
        af_classes = (af_classes - 1).type(torch.LongTensor)  # The targets should be in the range [0, 2], Pytorch requires
        pred_classes, af_classes = pred_classes.to(device), af_classes.to(device)
        if weights != None:
            #weights = weights.to(device)
            loss = F.cross_entropy(pred_classes, af_classes, weight=weights)
        else:
            loss = F.cross_entropy(pred_classes, af_classes)
        # Backward pass
        loss.backward()
        # Optimize
        optimizer.step()
        # Update
        bs = len(traces)
        total_loss += loss.detach().cpu().numpy()
        n_entries += bs
        # Update train bar
        train_bar.desc = train_desc.format(ep, total_loss / n_entries)
        train_bar.update(1)
    train_bar.close()
    return total_loss / n_entries


def eval(ep, dataload, weights):
    model.eval()
    total_loss = 0
    #total_diff = 0
    n_entries = 0
    all_logits = []
    all_labels = []
    eval_desc = "Epoch {:2d}: valid - Loss: {:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(dataload),
                    desc=eval_desc.format(ep, 0, 0), position=0)
    for traces, af_classes in dataload:
        traces = traces.transpose(1, 2)
        #print(traces.shape)
        #print(traces)
        #a = b
        traces, af_classes = traces.to(device), af_classes.to(device)
        with torch.no_grad():
            # Forward pass
            pred_classes = model(traces)
            # append
            all_logits.append(pred_classes.detach().cpu())
            all_labels.append(af_classes.detach().cpu())
            #loss = compute_loss(ages, pred_ages, weights)
            pred_classes = pred_classes.type(torch.DoubleTensor)  # float type raises error
            af_classes = (af_classes - 1).type(torch.LongTensor)
            pred_classes,af_classes = pred_classes.to(device), af_classes.to(device)
            if weights != None:
                #weights = weights.to(device)
                loss = F.cross_entropy(pred_classes, af_classes, weight=weights)
            else:
                loss = F.cross_entropy(pred_classes, af_classes)
            #print(loss)
            #a = b

            # Update outputs
            bs = len(traces)
            # Update ids
            total_loss += loss.detach().cpu().numpy()
            #total_diff += diff_w.detach().cpu().numpy()
            n_entries += bs
            # Print result
            eval_bar.desc = eval_desc.format(ep, total_loss / n_entries)
            eval_bar.update(1)
    eval_bar.close()
    # compute metrics
    metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
    # mean validation loss
    mean_valid_loss = total_loss / n_entries
    #print("ROC_AUC values: ", metrics[0])
    #print("Average precision values: ", metrics[1])

    return mean_valid_loss, metrics


if __name__ == "__main__":
    import h5py
    import pandas as pd
    import argparse
    from warnings import warn

    # Arguments that will be saved in config file
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Train model to predict rage from the raw ecg tracing.')
    parser.add_argument('--epochs', type=int, default=70,
                        help='maximum number of epochs (default: 70)')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed for number generator (default: 2)')
    parser.add_argument('--sample_freq', type=int, default=400,
                        help='sample frequency (in Hz) in which all traces will be resampled at (default: 400)')
    parser.add_argument('--seq_length', type=int, default=4096,
                        help='size (in # of samples) for all traces. If needed traces will be zeropadded'
                                    'to fit into the given size. (default: 4096)')
    parser.add_argument('--scale_multiplier', type=int, default=10,
                        help='multiplicative factor used to rescale inputs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument("--patience", type=int, default=7,
                        help='maximum number of epochs without reducing the learning rate (default: 7)')
    parser.add_argument("--min_lr", type=float, default=1e-7,
                        help='minimum learning rate (default: 1e-7)')
    parser.add_argument("--lr_factor", type=float, default=0.1,
                        help='reducing factor for the lr in a plateu (default: 0.1)')
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 192, 256, 320],
                        help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
    parser.add_argument('--net_seq_lengh', type=int, nargs='+', default=[4096, 1024, 256, 64, 16],
                        help='number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='dropout rate (default: 0.8).')
    parser.add_argument('--kernel_size', type=int, default=17,
                        help='kernel size in convolutional layers (default: 17).')
    parser.add_argument('--folder', default='model/',
                        help='output folder (default: ./out)')
    parser.add_argument('--traces_dset', default='signal',
                        help='traces dataset in the hdf5 file.')
    parser.add_argument('--ids_dset', default='id_exam',
                        help='by default consider the ids are just the order')
    parser.add_argument('--class_col', default='exam_class',
                        help='column with the exam classes in csv file.')
    parser.add_argument('--split_col', default='split',
                        help='column with train-test split info in csv file.')
    parser.add_argument('--ids_col', default='id_exam',
                        help='column with the ids in csv file.')
    parser.add_argument('--cuda', action='store_false',
                        help='use cuda for computations. (default: True)')
    parser.add_argument('--n_valid', type=int, default=-1,
                        help='the first `n_valid` exams in the hdf will be for validation.'
                             'The rest is for training')
    parser.add_argument('--use_weights', default= False,
                        help='Whether to use weights or not while training.')
    parser.add_argument('path_to_traces',
                        help='path to file containing ECG traces')
    parser.add_argument('path_to_csv',
                        help='path to csv file containing attributes.')
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    torch.manual_seed(args.seed)
    print(args)
    # Set device
    device = torch.device('cuda:0' if args.cuda else 'cpu')
    folder = args.folder

    # Generate output folder if needed
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    # Save config file
    with open(os.path.join(args.folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent='\t')

    tqdm.write("Building data loaders...")
    # Get csv data
    df = pd.read_csv(args.path_to_csv, index_col=args.ids_col)
    # 	ages = df[args.age_col]
    # Get h5 data
    f = h5py.File(args.path_to_traces, 'r')
    traces = f[args.traces_dset]
    if args.ids_dset:
        h5ids = f[args.ids_dset]
        df = df.reindex(h5ids, fill_value=False, copy=True)
    #ages = df[args.age_col]
    af_classes = df[args.class_col]
    # Train/ val split
    #valid_mask = np.arange(len(df)) <= args.n_valid
    #train_mask = ~valid_mask

    print("Number of samples: ", len(af_classes))

    # For testing the code: # Choose a small chunk of data for validation and training
    '''
    valid_mask_0 = np.arange(len(df)) < args.n_valid
    train_mask_0 = (~valid_mask_0) & (np.arange(len(df)) <= 10000)
    '''


    # For training code on full dataset "traces100pc" , can select few traces with n_valid
    # Can also use traces15pc without class 0 (pre-selected).

    if args.n_valid > 0:
        mask_chunk = np.arange(len(df)) < args.n_valid
    else:
        mask_chunk = np.arange(len(df)) < len(df)  # Full dataset

    train_mask_0 = (df[args.split_col] == "train").to_numpy()
    valid_mask_0 = (df[args.split_col] == "valid").to_numpy()

    train_mask = train_mask_0 & mask_chunk
    valid_mask = valid_mask_0 & mask_chunk


    # using CODE-15% dataset, --n_valid=105000
    '''
    mask_classes = (df[args.class_col] != 0).to_numpy()
    valid_mask_0 = (np.arange(len(df)) >= 70000) & (np.arange(len(df)) < args.n_valid)
    train_mask_0 = np.arange(len(df)) >= args.n_valid

    valid_mask = valid_mask_0 & mask_classes
    train_mask = train_mask_0 & mask_classes
    '''

    # Get an array of class values for training
    af_classes_train = af_classes[train_mask].to_numpy()
    print(af_classes_train[:20], "Number of traces used for training: ", len(af_classes_train))
    af_classes_valid = af_classes[valid_mask].to_numpy()
    print(af_classes_valid[:20], "Number of traces used for validation: ", len(af_classes_valid))

    # Compute rescaling weights
    if args.use_weights:
        weights = compute_weights(af_classes_train)
        weights = torch.tensor(weights)
        print("Weights values: ", weights)
    else:
        weights = None
    #a=b
    print("Number of samples", len(train_mask))
    # weights
    # weights = compute_weights(ages)

    # Dataloader
    train_loader = BatchDataloader(traces, af_classes, bs=args.batch_size, mask=train_mask)
    valid_loader = BatchDataloader(traces, af_classes, bs=args.batch_size, mask=valid_mask)
    tqdm.write("Done!")

    tqdm.write("Define model...")
    N_LEADS = 12  # the 12 leads
    N_CLASSES = 3  # Classes 1, 2 and 3
    model = ResNet1d(input_dim=(N_LEADS, args.seq_length),
                     blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh)),
                     n_classes=N_CLASSES,
                     kernel_size=args.kernel_size,
                     dropout_rate=args.dropout_rate)
    model.to(device=device)
    tqdm.write("Done!")

    tqdm.write("Define optimizer...")
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=5e-4)
    tqdm.write("Done!")

    tqdm.write("Define scheduler...")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience,
                                                     min_lr=args.lr_factor * args.min_lr,
                                                     factor=args.lr_factor)
    tqdm.write("Done!")

    tqdm.write("Training...")
    start_epoch = 0
    best_loss = np.Inf
    history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr',
                                    'AUC_class1', 'AUC_class2', 'AUC_class3',
                                    'AP_class1', 'AP_class2', 'AP_class3',
                                    'AUC_3to1', 'AP_3to1'])
    for ep in range(start_epoch, args.epochs):
        train_loss = train(ep, train_loader, weights)
        valid_loss, metrics = eval(ep, valid_loader, weights)

        # Save best model
        if valid_loss < best_loss:
            # Save model
            torch.save({'epoch': ep,
                        'model': model.state_dict(),
                        'valid_loss': valid_loss,
                        'optimizer': optimizer.state_dict()},
                       os.path.join(folder, 'model.pth'))
            # Update best validation loss
            best_loss = valid_loss
        # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        # Interrupt for minimum learning rate
        if learning_rate < args.min_lr:
            break
        # Print message
        tqdm.write('Epoch {:2d}: \tTrain Loss {:.6f} ' \
                  '\tValid Loss {:.6f} \tLearning Rate {:.7f}\t'
                 .format(ep, train_loss, valid_loss, learning_rate))
        # Save history
        history = history.append({"epoch": ep, "train_loss": train_loss, "valid_loss": valid_loss,
                                  "lr": learning_rate, "AUC_class1": metrics[0][0],
                                  "AUC_class2": metrics[0][1],"AUC_class3": metrics[0][2],
                                  "AP_class1": metrics[1][0], "AP_class2": metrics[1][1],
                                  "AP_class3": metrics[1][2], "AUC_3to1": metrics[2][0],
                                  "AP_3to1": metrics[2][1]}, ignore_index=True)

        history.to_csv(os.path.join(folder, 'history.csv'), index=False)
        # Update learning rate
        scheduler.step(valid_loss)

    tqdm.write("Done!")


import pandas as pd
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description='Compare two csv file.', add_help=False)
    parser.add_argument('originalfile', type=str,
                        help='path to original csv file ')
    parser.add_argument('newfile', type=str,
                        help='path to new csv file')
    parser.add_argument('--ids_original', default='exam_ids',
                        help='names of column contain exam ids (original file).')
    parser.add_argument('--ids_new', default='exam_ids',
                       help='names of column contain exam ids (original file).')
    parser.add_argument('--tol', type=float, default=1e-6,
                        help='tolerance in the comparison.')

    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")
    # load
    original = pd.read_csv(args.originalfile)
    new = pd.read_csv(args.newfile)

    # Match ids
    ids_original = original[args.ids_original].values
    ids_new = new[args.ids_new].values

    # Get intersection
    ids_intersection = np.intersect1d(ids_original, ids_new)

    # Set ids as index
    original.set_index(args.ids_original, inplace=True)
    new.set_index(args.ids_new, inplace=True)

    #  Set intersection as index
    original = original.loc[ids_intersection]
    new = new.loc[ids_intersection]

    # Compare prob_classi, i = 1, 2, 3
    all_equal = True
    for i in range(1, 4):
        if np.max(original[f'prob_class{i}'] - new[f'prob_class{i}']) > args.tol:
            all_equal = False
    print(f'Predicted probabilities are equal: {all_equal}')

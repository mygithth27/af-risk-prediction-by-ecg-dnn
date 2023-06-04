import pandas as pd
import numpy as np
import argparse
import warnings
import os
from datetime import date as dt

N_GROUPS = 3
N_CLASSES = 3
CLASSES = ['no_af', 'af', 'will_develop_af']


def get_extended_dataframe(path_to_csv, min_deltatime, max_deltatime):
    # Read
    df = pd.read_csv(path_to_csv)
    # Sort by date
    df.sort_values('date_exam', inplace=True, ascending=False)
    n_exams = len(df)

    # Get some fields
    ids = np.array(df['id_exam'])
    patient_ids = np.array(df['id_patient'].astype(int))
    date = pd.to_datetime(df['date_exam']).values
    condition = np.array(df['AF'], dtype=bool)

    # Get number of patients
    patients = np.unique(patient_ids)
    n_patients = len(patients)
    # Get converters
    hash_exams = dict(zip(ids, range(n_exams)))
    hash_patients = dict(zip(patients, range(n_patients)))

    # Get information about next exam (notice the exams are ordered in descending order)
    next_exam_id = -1 * np.ones(n_exams, dtype=int)
    count_exams = np.zeros(n_patients, dtype=int)
    date_last_exam = np.zeros(n_patients, dtype='datetime64[ns]')
    first_exam_patient = -1 * np.ones(n_patients, dtype=int)
    patients_with_condition = np.zeros(n_patients, dtype=bool)
    for n in range(n_exams):
        n_patient = hash_patients[patient_ids[n]]
        next_exam_id[n] = first_exam_patient[n_patient] # Same patient, next exam id for this Nth exam (desc. date order)
        patients_with_condition[n_patient] = patients_with_condition[n_patient] or condition[n]
        if first_exam_patient[n_patient] > 0:
            n_next = hash_exams[first_exam_patient[n_patient]]
        else:
            date_last_exam[n_patient] = date[n]
        first_exam_patient[n_patient] = ids[n]  # First exam id per patient
        count_exams[n_patient] += 1

    # First appearance
    count_appearances_exam = np.zeros(n_exams, dtype=int)
    count_appearances = np.zeros(n_patients, dtype=int) # Number of positive results of AF per patient
    crescent_counter = np.zeros(n_patients, dtype=int)  # The number of exams per patient
    count_exams_first_appearance = np.zeros(n_patients, dtype=int)
    date_first_appearance = np.zeros(n_patients, dtype='datetime64[ns]')
    for n in range(n_exams)[::-1]:
        n_patient = hash_patients[patient_ids[n]]
        crescent_counter[n_patient] += 1
        if condition[n]:
            count_appearances[n_patient] += 1
        count_appearances_exam[n] = count_appearances[n_patient]    # The number of AF positive results at current exam for the patient
        if condition[n] and count_exams_first_appearance[n_patient] == 0:
            count_exams_first_appearance[n_patient] = crescent_counter[n_patient] # The nth exam at which AF appeared for the patient
            date_first_appearance[n_patient] = date[n]

    # Divide patients in three groups:
    # group 0 - not belong to any of the groups
    # group 1 - more than one exam, never develop condition.
    # group 2 - condition on the first exam.
    # group 3 - will develop condition (multiple exams)
    group = np.zeros((n_patients,), dtype=int)
    group[(count_exams >= 2) & ~patients_with_condition] = 1
    group[patients_with_condition & (count_exams_first_appearance == 1)] = 2
    group[patients_with_condition & (count_exams_first_appearance > 1)] = 3

    # Get the group of each exam
    group_exam = np.zeros(n_exams, dtype=int)
    for n in range(n_exams):
        n_patient = hash_patients[patient_ids[n]]
        group_exam[n] = group[n_patient]

    # Divide exams in 3 classes:
    # class 0: not belong to any class
    # class 1: will never develop condition and has more than one exam
    #   i.e. all exams in group 1 that have a next exam
    # class 2: display the condition
    #   i.e. all the first exams in group 2 (+ 1st appearance in group 3)
    # class 3: Patients with more than one exam that
    #   develop AF after the first exam - we consider
    #   all the exams before the AF respecting the given
    #   time span to be in class 3.
    exam_class = np.zeros((n_exams,), dtype=int)

    # Get time before last exam for group 1
    time_to_last_exam = np.zeros(n_exams, dtype='timedelta64[ns]')
    for n in range(n_exams):
        if (group_exam[n] == 1) & (next_exam_id[n] != -1):
            n_patient = hash_patients[patient_ids[n]]
            time_to_last_exam[n] = date_last_exam[n_patient] - date[n]

    # Convert to weeks
    time_to_last_exam = (time_to_last_exam / ((1e9) * 60 * 60 * 24 * 7)).astype(int)

    # Get class 1
    exam_class[(group_exam == 1) & (next_exam_id != -1)
               & (min_deltatime < time_to_last_exam) & (time_to_last_exam < max_deltatime)] = 1

    # Get class 2
    exam_class[condition] = 2

    # Get time before first apperance
    time_to_first_appearance = np.zeros(n_exams, dtype='timedelta64[ns]')
    for n in range(n_exams):
        if (group_exam[n] == 3) & (count_appearances_exam[n] < 1):
            n_patient = hash_patients[patient_ids[n]]
            time_to_first_appearance[n] = date_first_appearance[n_patient] - date[n]

    # Convert to weeks
    time_to_first_appearance = (time_to_first_appearance / ((1e9) * 60 * 60 * 24 * 7)).astype(int)

    # Get class 3
    exam_class[(group_exam == 3) & (count_appearances_exam < 1) &
               (min_deltatime < time_to_first_appearance) & (time_to_first_appearance < max_deltatime)] = 3

    # Get patients without any class
    patients_not_in_class = np.ones((n_patients, N_CLASSES), dtype=bool)    # [[1,1,1]]: no class or class '0'
    for n in range(n_exams):
        for c in range(N_CLASSES):
            if exam_class[n] == c + 1:
                n_patient = hash_patients[patient_ids[n]]
                patients_not_in_class[n_patient, c] = 0

    # Save all the information about exams in df
    df['group_exam'] = group_exam
    df['exam_class'] = exam_class
    df['time_to_first_appearance'] = time_to_first_appearance # for exams in group 3 & before appearence
    df['time_to_last_exam'] = time_to_last_exam     # for exams in group 1 & not last

    # Create dataframe containing information about patient
    d = {'patient_ids': patients, 'group': group}   # Per unique and sorted patient Ids
    d.update({'not_in_class_{}'.format(c): patients_not_in_class[:, c] for c in range(N_CLASSES)})
    df_patient = pd.DataFrame(d)

    print("Time to last exam:\n", time_to_last_exam[-100:])
    print("Time to first appearence:\n", time_to_first_appearance[-200:])

    return df, df_patient


def get_num_patients_per_group(df, df_patient):
    """Return n_groups  array containing how many patients in each group"""
    num_patient_group = np.zeros((N_GROUPS,), dtype=int)
    for group in range(N_GROUPS):
            num_patient_group[group] = sum(df_patient['group'] == group + 1)
    return num_patient_group


def get_num_exams_per_group(df, _df_patient):
    """Return n_groups array containing how many exams in each group"""
    num_exams_group = np.zeros((N_GROUPS,), dtype=int)
    for group in range(N_GROUPS):
            num_exams_group[group] = sum(df['group_exam'] == group + 1)
    return num_exams_group


def get_num_patients_per_group_per_class(df, df_patient):
    """Return n_groups x n_classes array containing how many patients in each group and class"""
    num_patient_group_class = np.zeros((N_GROUPS, 3), dtype=int)
    for group in range(N_GROUPS):
        for c in range(N_CLASSES):
            num_patient_group_class[group, c] = \
                sum((df_patient['group'] == group + 1) & ~df_patient['not_in_class_{}'.format(c)])
    return num_patient_group_class


def get_num_exams_per_group_per_class(df, df_patient):
    """Return n_groups x n_classes array containing how many ECG in each group and class"""
    num_group_class = np.zeros((N_GROUPS, 3), dtype=int)
    for group in range(N_GROUPS):
        for c in range(N_CLASSES):
            num_group_class[group, c] = sum((df['group_exam'] == group + 1) & (df['exam_class'] == c + 1))
    return num_group_class


def print_summary(df, df_patient):
    """Print summary for number of patients in each group and exams in each class"""
    num_patient_group_class = get_num_patients_per_group_per_class(df, df_patient)
    num_patient_group = get_num_patients_per_group(df, df_patient)
    num_exams_group_class = get_num_exams_per_group_per_class(df, df_patient)
    num_exams_group = get_num_exams_per_group(df, df_patient)

    for group in range(N_GROUPS):
        print("Group {:1d} = {:7d} exams - {:6d} unique patients".format(group+1, num_exams_group[group],
                                                                         num_patient_group[group]))
        for c in range(N_CLASSES):
            if num_exams_group_class[group, c] != 0:
                print("          class {:1d}: {:7d} exams in  - {:6d} unique patients.".format(
                    c+1, num_exams_group_class[group, c], num_patient_group_class[group, c]))


def get_patient_ids(df_patient, valid_split, test_split, seed=0):
    """Return train, validation and test patient ids"""

    # Initialize state
    rng = np.random.RandomState(seed)

    # Get patients from group 1 class 1
    patients_g1_c1 = list(df_patient.loc[(df_patient['group'] == 1) & ~df_patient['not_in_class_{}'.format(0)], 'patient_ids'])  # returning patient's id, not index
    
    # Get patients from group 2 class 2
    patients_g2_c2 = list(df_patient.loc[(df_patient['group'] == 2) & ~df_patient['not_in_class_{}'.format(1)], 'patient_ids'])  #  returning patients' ids
    
    # Get patients from group 3 class 2 (and not in class 3)
    patients_g3_c2 = list(df_patient.loc[(df_patient['group'] == 3) & ~df_patient['not_in_class_{}'.format(1)]
        & df_patient['not_in_class_{}'.format(2)], 'patient_ids'])  # returning patients' ids

    # Get patients from group 3 class 3
    patients_g3_c3 = list(df_patient.loc[(df_patient['group'] == 3) & ~df_patient['not_in_class_{}'.format(2)], 'patient_ids']) # returning patients' ids

    train_ids, valid_ids, test_ids = [], [], []

    in_all = [patients_g1_c1, patients_g3_c3]       # All-exams-normal patients and patients who develop later AF and in class 3

    for p in in_all:
        length = len(p)
        valid_length = int(np.floor(valid_split * length))
        test_length = int(np.floor(test_split * length))
        rng.shuffle(p)
        train_ids += p[valid_length + test_length:]
        valid_ids += p[test_length:valid_length+test_length]
        test_ids += p[:test_length]

    in_train_and_valid = [patients_g2_c2, patients_g3_c2]   # First-exam-positive patients and patients who develop later AF
    for p in in_train_and_valid:
        length = len(p)
        valid_length = int(np.floor(valid_split * length))
        rng.shuffle(p)
        train_ids += p[valid_length:]
        valid_ids += p[:valid_length]

    return train_ids, valid_ids, test_ids


def get_ids(df, train_patient_ids, valid_patient_ids, test_patient_ids):
    """Return train, validation and test exam ids"""
    # Get patient ids
    patient_ids = [train_patient_ids, valid_patient_ids, test_patient_ids]
    # Get all ids in classes {1, 2, 3} for the patient ids
    ids = []
    for p_ids in patient_ids:
        ixs = df["id_patient"].isin(p_ids) & (df['exam_class'] != 0) & (df['group_exam'] != 0)  # index mask
        ex_ids = list(df.loc[ixs, 'id_exam'])
        ids.append(ex_ids)
    return ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate data summary for the AF prediction problem')
    parser.add_argument('file', default='annotations.csv',
                        help='csv file to read data from (default:./annotations.csv)')
    parser.add_argument('--out_folder', default='out',
                        help='csv file to read data from (default:./annotations.csv)')
    parser.add_argument('--min_deltatime', default=0, type=int,
                        help='min time between the ECG exam and the exam which display the condition (in weeks)'
                             'for the exam to be included in class 3. Default is: 0')
    parser.add_argument('--max_deltatime', default=1000, type=int,
                        help='max time between the ECG exam and the exam which display the condition (in weeks)'
                             'for the exam to be included in class 3. Default is: 0')
    parser.add_argument('--silent', action='store_true',
                        help='dont generate summary of the data')
    parser.add_argument('--skip_plots', action='store_true',
                        help='dont generate plots')
    parser.add_argument('--test_split', default=0.3, type=float,
                        help='Number between 0 and 1. Percentage of the data to be used in the test split.')
    parser.add_argument('--valid_split', default=0.1, type=float,
                        help='Number between 0 and 1. Percentage of the data to be used in the validation split.')
    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")

    # Get dataframe with info about exams and about patients
    exams_info, patients_info = get_extended_dataframe(args.file, args.min_deltatime,  args.max_deltatime)
    
    # Divide patient ids into train, valid and test
    train_patient_ids, valid_patient_ids, test_patient_ids = get_patient_ids(patients_info, args.valid_split,
                                                                                args.test_split, seed=0)
    set1 = set(train_patient_ids) 
    common_elements = set1.intersection(test_patient_ids)
    print("test_patient ids: ", len(test_patient_ids))
    print("Common p_ids train-test: ", len(common_elements))
    
    patients_info['split'] = 'none'
    patients_info.loc[np.isin(patients_info['patient_ids'], train_patient_ids), 'split'] = 'train'
    patients_info.loc[np.isin(patients_info['patient_ids'], valid_patient_ids), 'split'] = 'valid'
    patients_info.loc[np.isin(patients_info['patient_ids'], test_patient_ids), 'split'] = 'test'

    # Divide exam ids into train, valid and test
    train_exam_ids, valid_exam_ids, test_exam_ids = get_ids(exams_info, train_patient_ids,
                                                            valid_patient_ids, test_patient_ids)
        
    set2 = set(train_exam_ids)
    common_ids = set2.intersection(test_exam_ids)
    print("test_exam ids: ", len(test_exam_ids))
    print("Common exam_ids train-test: ", len(common_ids))
    
    exams_info['split'] = 'none'
    exams_info.loc[np.isin(exams_info['id_exam'], train_exam_ids), 'split'] = 'train'
    exams_info.loc[np.isin(exams_info['id_exam'], valid_exam_ids), 'split'] = 'valid'
    exams_info.loc[np.isin(exams_info['id_exam'], test_exam_ids), 'split'] = 'test'

    # Save
    if not os.path.isdir(args.out_folder):
        os.mkdir(args.out_folder)
    exams_info.to_csv(os.path.join(args.out_folder, 'exams_info.csv'), index=False)
    patients_info.to_csv(os.path.join(args.out_folder, 'patients_info.csv'), index=False)

    # Print info
    if not args.silent:
        print_summary(exams_info, patients_info)

    # Generate plots
    if not args.skip_plots:
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Generate plots
        fig, ax = plt.subplots(ncols=2)
        sns.histplot(data=exams_info['time_to_first_appearance'], kde=False,  bins=np.logspace(-1, 3, 10), ax=ax[0]) # or seaborn.displot
        ax[0].set(xscale='log',)
        ax[0].set(xticks=[0.1, 1, 10, 100, 1000])
        ax[0].set_xticklabels(['16 hours', '1 week', '3 months', '2 years', '20 years'], rotation=40)
        ax[0].set_title('time before first appearance (group 3)')
        sns.histplot(data=exams_info['time_to_last_exam'], kde=False, bins=np.logspace(-1, 3, 10), ax=ax[1])
        ax[1].set(xscale='log',)
        ax[1].set(xticks=[0.1, 1, 10, 100, 1000])
        ax[1].set_xticklabels(['16 hours', '1 week', '3 months', '2 years', '20 years'], rotation=40)
        ax[1].set_title('time before last exam(group 1)')
        plt.show()
    

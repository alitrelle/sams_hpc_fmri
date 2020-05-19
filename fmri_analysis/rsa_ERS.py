# code written by S Guerin and updated by A Trelle #

from glob import glob
from os.path import exists
from mvpa_functions import *
from itertools import product
from nilearn.input_data import NiftiMasker
import nibabel as nib
from numpy.linalg import inv


def main():
    #subs2run = list(pd.read_csv('SH_CR.csv').iloc[:,1])
    subs2run = list(pd.read_csv('subs_paper1_final.csv').iloc[:,1])

    # Initialize conditions;
    conditions = list(['Source_Hit','Non_Source_Hit'])

    for subject in subs2run:
        roi_name = 'ANG'
        roi_file = '/scratch/users/atrelle/AM/data/{}/masks/YeoANG.nii.gz'.format(subject)

        if exists(roi_file):

            for condition in conditions:

                if not exists('./paper_one/rsa_ers_TR/{}/rsa_ers_{}_{}.csv'.format(condition,roi_name,subject)):

                    # Handle files and load BOLD data
                    # ----------------------------------------------------------------------
                    files = manage_files(subject,roi_file)

                    # We want to load the study and test data together in one matrix (and
                    # then split) to ensure that masking is equivalent for the two (out of
                    # brain voxels are excluded by load_data by default)
                    print('Loading bold data for subject {}'.format(subject))
                    big_X, big_X_groups, original_mask_cols = load_data(files['bold_files'],
                                                                        files['roi_file'])

                    # Estimate regression modeling effects of head motion and artifacts on
                    # fMRI time series and get residuals
                    # ----------------------------------------------------------------------
                    DM = get_dm(big_X_groups, motion_files=files['motion_files'],
                            artifact_files=files['artifact_files'])

                    betas, big_X_resid = glm(DM, big_X)

                    # Split up the data into study and test components
                    big_X_study, big_X_test, big_X_groups_study, big_X_groups_test = \
                        split_data(big_X_resid, big_X_groups, files)

                    # Get cleaned up onsets (drops trials with artifacts)
                    # ----------------------------------------------------------------------
                    study_onsets_df, test_onsets_df = get_onsets(files,condition)

                    # Average peak for each trial
                    # ----------------------------------------------------------------------
                    # X_study_all_trials = avg_peak(big_X_study, study_onsets_df, peak_TRs=[2, 3, 4])
                    # X_test_all_trials = avg_peak(big_X_test, test_onsets_df,
                    #                              peak_TRs=[3, 4, 5])

                    X_study_all_trials = avg_peak(big_X_study, study_onsets_df, peak_TRs=[2, 3, 4])
                    X_test_all_trials = avg_peak(big_X_test, test_onsets_df,
                                                 peak_TRs=[3, 4, 5])

                    # Determine common trials for encoding-retreval similarity

                    # Add labels to trial x voxel matrix for study and test
                    X_test_df = pd.DataFrame(data = X_test_all_trials)
                    X_test_labeled = pd.concat([test_onsets_df,X_test_df],axis = 1)

                    X_study_df = pd.DataFrame(data = X_study_all_trials)
                    X_study_labeled = pd.concat([study_onsets_df,X_study_df],axis = 1)

                    # Remove 'other' items from test onsets and X_test.
                    X_test_labeled = X_test_labeled.loc[X_test_labeled['condition'] !='OTHER',:]

                    # Merge test data with the study data to yield common trials only
                    combined_df_test = pd.merge(study_onsets_df,X_test_labeled,on = 'word')

                    # Split trial x voxel matrix from onset data
                    cols = list(np.arange(X_test_all_trials.shape[1]))
                    X_test = np.array(combined_df_test.loc[:,cols])

                    new_study_df = combined_df_test[['word','cat_x','concat_onset_x','condition_y','run_x']]
                    new_test_df = combined_df_test[['word','cat_y','concat_onset_y','condition_y','run_y','RT_y']]

                    combined_df_test = combined_df_test[['word','cat_y','concat_onset_y','condition_y','run_y']]

                    combined_df_study = pd.merge(combined_df_test,X_study_labeled,on = 'word')
                    cols = list(np.arange(X_study_all_trials.shape[1]))
                    X_study = np.array(combined_df_study.loc[:,cols])

                    # RSA estimation
                    # ----------------------------------------------------------------------
                    # Study labels & data
                    y_cat1 = new_study_df['cat_x']
                    y_item1 = new_study_df['word']
                    X1 = X_study

                    # Test labels & data
                    y_cat2 = new_test_df['cat_y']
                    y_item2 = new_test_df['word']
                    y_cond = new_test_df['condition_y']
                    y_RT = new_test_df['RT_y']
                    X2 = X_test

                    df = RSA(y_cat1, y_item1, X1, y_cat2, y_item2, y_cond, y_RT, X2)

                    df.to_csv('./paper_one/rsa_ers_TR/{}/rsa_ers_{}_{}.csv'.format(condition,roi_name,subject))

def manage_files(subject,roi_file):

    # #labels2include = ['fusiform','inferiortemporal'] #can comment out
    # roi_mask_name = 'VTC_fusi_inftemp.nii.gz' #remove .gz from name
    # folder = 'masks' #change folder as needed to T2_seg
    # SUBJECTS_DIR = '/share/awagner/AM/data'

    # Get combined mask if file does not already exist
    # roi_file = '{}/{}/{}/{}'.format(SUBJECTS_DIR,subject,folder,roi_mask_name)
    # roi_file = '/share/awagner/AM/data/{}/masks/{}'.format(subject,
    #     roi_mask_name)

    # if not exists(roi_file):
    #     print('Creating freesurfer mask for subject {}'.format(subject))
    #     write_masks(subject, labels2include, roi_mask_name, SUBJECTS_DIR,
    #         exp='study')

    # Get paths to files we need
    bold_files = sorted(glob(('/scratch/users/atrelle/AM/analysis/*/{}/reg/epi/'
        + 'unsmoothed/run_*/timeseries_xfm.nii.gz').format(subject)))

    study_bold_files = sorted(glob(('/scratch/users/atrelle/AM/analysis/study/{}/reg/'
        + 'epi/unsmoothed/run_*/timeseries_xfm.nii.gz').format(subject)))

    test_bold_files = sorted(glob(('/scratch/users/atrelle/AM/analysis/test/{}/reg/epi/'
        + 'unsmoothed/run_*/timeseries_xfm.nii.gz').format(subject)))

    study_onsets_file = \
        '/oak/stanford/groups/awagner/AM/data/{}/behav/study_rsa_RT.csv'.format(subject)
    test_onsets_file = \
        '/oak/stanford/groups/awagner/AM/data/{}/behav/test_rsa_RT.csv'.format(subject)

    # Get motion & artifact files for all study and test runs (glob)
    motion_files = sorted(glob(('/scratch/users/atrelle/AM/analysis/*/{}/preproc/run_*/'
        + 'realignment_params.csv').format(subject)))

    artifact_files = sorted(glob(('/scratch/users/atrelle/AM/analysis/*/{}/preproc/'
        + 'run_*/artifacts.csv').format(subject)))

    study_artifact_files = sorted(glob(('/scratch/users/atrelle/AM/analysis/study/{}/'
        + 'preproc/run_*/artifacts.csv').format(subject)))

    test_artifact_files = sorted(glob(('/scratch/users/atrelle/AM/analysis/test/{}/'
        + 'preproc/run_*/artifacts.csv').format(subject)))

    n_study_runs = len(study_bold_files)
    n_test_runs = len(test_bold_files)

    files = {'roi_file' : roi_file,
             'bold_files' : bold_files,
             'study_bold_files' : study_bold_files,
             'test_bold_files' : test_bold_files,
             'study_onsets_file' : study_onsets_file,
             'test_onsets_file' : test_onsets_file,
             'motion_files' : motion_files,
             'artifact_files' : artifact_files,
             'study_artifact_files' : study_artifact_files,
             'test_artifact_files' : test_artifact_files,
             'n_study_runs' : n_study_runs,
             'n_test_runs' : n_test_runs}

    return files


def split_data(big_X_resid, big_X_groups, files):
    n_study_runs = files['n_study_runs']
    n_test_runs = files['n_test_runs']

    # Split data into study and test, use residuals from regression model
    big_X_study = big_X_resid[big_X_groups <= n_study_runs, :]
    big_X_test = big_X_resid[big_X_groups > n_study_runs, :]

    big_X_groups_study = big_X_groups[big_X_groups <= n_study_runs]
    big_X_groups_test = big_X_groups[big_X_groups > n_study_runs]

    return big_X_study, big_X_test, big_X_groups_study, big_X_groups_test


def get_onsets(files,condition):
    # Handle onsets
    study_onsets_ALL_df = concat_onsets(files['study_onsets_file'], \
        files['study_bold_files'])
    study_onsets_df = remove_artifact_trials(study_onsets_ALL_df, \
        files['study_artifact_files'])

    test_onsets_ALL_df = concat_onsets(files['test_onsets_file'], \
        files['test_bold_files'])
    test_onsets_df = remove_artifact_trials(test_onsets_ALL_df, \
        files['test_artifact_files'])

    # Recode condition labels for test_onsets_df to match study
    # Note that there is considerable flexibility here in the combinations of
    # trial types you can include
    if condition == 'Source_Hit':
        test_labels = {'WP' : ['SHP'],
                       'WF' : ['SHF']}
    elif condition == 'Non_Source_Hit':
        test_labels = {'WP' : ['SMP','IP','MP'],
                       'WF' : ['SMF','IF','MF']}


        test_onsets_df['orig_cond'] = test_onsets_df['condition']
    for index, row in test_onsets_df.iterrows():
        if row['condition'] in test_labels['WP']:
            test_onsets_df.loc[index, 'condition'] = test_onsets_df.loc[index, 'condition']
        elif row['condition'] in test_labels['WF']:
            test_onsets_df.loc[index, 'condition']  = test_onsets_df.loc[index, 'condition']
        else:
            # new items & source miss or error
            test_onsets_df.loc[index, 'condition']  = 'OTHER'

    return study_onsets_df, test_onsets_df

def RSA(y_cat1, y_item1, X1, y_cat2, y_item2, y_cond, y_RT, X2):
    """Calculates correlations for a representational similarity analysis (RSA).

    All correlations have been transformed to Fisher's Z

    Returns
    ----------
    Dataframe containing within-event ERS (item-item), within-category ERS,
    and between-category ERS, as well as category, item, and condition info for each trial

    """
    # Calculate correlation matrix
    r_matrix = np.corrcoef(X1, X2)

    # If X2 is given as input, remove portions of correlation matrix
    # corresponding to X1, X1 correlations. (i.e., take upper right quadrant).
    r_matrix = r_matrix[0:X1.shape[0], X1.shape[0]:]

    # Verify that y_item1 and y_item2 are identical
    if not np.array_equal(y_cat1, y_cat2):
        raise ValueError('y_item1 and y_item2 are not identical sets')

    # Initialize
    df = pd.DataFrame(columns=['cat', 'item', 'cond', 'item-item z','within-cat z', 'between-cat z'])
    df['cat'], df['item'], df['cond'], df['RT'] = y_cat1, y_item1,y_cond,y_RT

    temp_df = pd.DataFrame(columns=['cat', 'item', 'cond', 'within-cat z_enc', 'between-cat z_enc',
                                'within-cat z_retr', 'between-cat z_retr'])

    # item-item (diagonal elements)
    n_rows = r_matrix.shape[0]

    # df['item-item z'] = np.arctanh(r_matrix[range(n_rows), range(n_rows)])
    df['item-item z'] = np.arctanh(np.diag(r_matrix))

    # Mask diagonal and all cells below (set to NaN, not included in mean)
    np.fill_diagonal(r_matrix,np.nan)

    # Calculate within and between category means. Base on between-run correlations only
    # if groups is provided as input.
    face_positions = []
    place_positions =[]
    for col in xrange(len(y_cat1)):
        if y_cat1[col] == 'face':
            face_positions.append(col)
        else:
            place_positions.append(col)

    for row in xrange(n_rows):
        if y_cat1[row] == 'face':
            df.loc[row, 'within-cat z'] = np.nanmean(np.arctanh(r_matrix[row,face_positions]))
            df.loc[row, 'between-cat z'] = np.nanmean(np.arctanh(r_matrix[row,place_positions]))
        else:
            df.loc[row, 'within-cat z'] = np.nanmean(np.arctanh(r_matrix[row,place_positions]))
            df.loc[row, 'between-cat z'] = np.nanmean(np.arctanh(r_matrix[row,face_positions]))

    return df

if __name__ == "__main__":
    main()

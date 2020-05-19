# code written by S Guerin and updated by A Trelle #

from create_combined_masks import combine_masks
from glob import glob
from os.path import exists
from mvpa_functions import *

def main():
    subs2run = list(pd.read_csv('subs_paper1_final.csv').iloc[:,1])
    standardize_across_voxels=False
    standardize_across_trials=True

    row=0
    for subject in subs2run:
        print(subject)

        # Handle files and load BOLD data
        # ----------------------------------------------------------------------
        condition = 'All_Conds'
        roi_name = 'ANG'
        roi_file = '/scratch/users/atrelle/AM/data/{}/masks/YeoANG.nii.gz'.format(subject)
        output_file='./paper_one/mvpa_retrieval/train_enc_test_rec_{}_{}_prob.csv'.\
            format(subject,roi_name)

        if not exists(output_file):


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

            # Get category labels for machine learning classification (drops trials
            # with artifacts)
            # ----------------------------------------------------------------------
            study_onsets_df, test_onsets_df = get_onsets(files,condition)
            y_study = study_onsets_df.condition
            groups_study = study_onsets_df.run

            # Average peak for each trial
            # ----------------------------------------------------------------------
            X_study = avg_peak(big_X_study, study_onsets_df, peak_TRs=[2, 3, 4])
            X_test_all_trials = avg_peak(big_X_test, test_onsets_df,
                                         peak_TRs=[2, 3, 4])

            # Remove 'other' items from test onsets and X_test. These items won't contribute to z-scoring
            X_test = X_test_all_trials[
                np.array(test_onsets_df['condition']!='OTHER'), :]

            y_test = test_onsets_df.loc[test_onsets_df['condition']!='OTHER',
                'condition'].reset_index(drop=True)
            groups_test = test_onsets_df.loc[test_onsets_df['condition']!='OTHER',
                'run'].reset_index(drop=True)
            y_test_cond = test_onsets_df.loc[test_onsets_df['condition']!='OTHER',
                'orig_cond'].reset_index(drop=True)

            # get uni and t-stat from selected voxels
            y_study = get_binary_y(y_study)

            voxels2use,cat0_voxels,cat1_voxels,cat0_tvals, cat1_tvals = \
                balanced_feat_sel(X_study, y_study, tot_num_vox=500)

            X_test_face = X_test[:,cat0_voxels].mean(axis=1)
            X_test_place = X_test[:,cat1_voxels].mean(axis=1)
            X_test_all = X_test[:,voxels2use].mean(axis=1)
            X_test_orig = X_test.mean(axis=1)

            #revert back to non-binary
            y_study = study_onsets_df.condition

            # if z-score across voxels
            if standardize_across_voxels==True:
                # z-score across voxels for each trial
                X_study = zscore(X_study,axis=1)
                X_test = zscore(X_test,axis=1)

            # if zscore across trials (within run)
            if standardize_across_trials==True:

                for run in pd.unique(study_onsets_df['run']):
                    # z-score across trials (within run)
                    X_study[(groups_study==run),:] = zscore(X_study[(groups_study==run),:],axis=0)

                for run in pd.unique(test_onsets_df['run']):
                    # z-score across trials (within run)
                    X_test[(groups_test==run),:] = zscore(X_test[(groups_test==run),:],axis=0)

            # L2 Logistic Regression Classifier (train study, test recall)
            # ----------------------------------------------------------------------
            print('Train encoding, test recall for subject {}'.format(subject))
            auc_df, prob_df = LogReg_Permutation(X_train=X_study, X_test=X_test,
                y_train=y_study, y_test=y_test, train_groups=groups_study,
                tot_num_vox = 500, n_permutations=0,subsample_iterations=10)

            # Add fields normally added by CV_LogReg_Permutation:
            prob_df.loc[:, 'Run'] = np.array(groups_test)
            prob_df.loc[:, 'Category'] = np.array(y_test)
            prob_df.loc[:,'Condition'] = np.array(y_test_cond)
            prob_df.loc[:,'Word'] = np.array(test_onsets_df.loc[
                test_onsets_df['condition']!='OTHER', 'word'])
            prob_df.loc[:, 'Onset'] = np.array(test_onsets_df.loc[
                test_onsets_df['condition']!='OTHER', 'concat_onset'])
            prob_df.loc[:,'RT'] = np.array(test_onsets_df.loc[
                test_onsets_df['condition']!='OTHER', 'RT'])

            prob_df.loc[:,'Face_Uni'] = X_test_face
            prob_df.loc[:,'Place_Uni'] = X_test_place
            prob_df.loc[:,'Sel_Uni'] = X_test_all
            prob_df.loc[:,'Uni'] = X_test_orig

            # Save output files

            prob_df.to_csv('./paper_one/mvpa_retrieval/train_enc_test_rec_{}_{}_prob.csv'.\
                format(subject,roi_name))


    return None

def manage_files(subject,roi_file):

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
    if condition == 'All_Conds':
        test_labels = {'WP' : ['SHP','SMP','IP','MP'],
                       'WF' : ['SHF','SMF','IF','MF']}
    elif condition =='Source_Hit':
        test_labels = {'WP' : ['SHP'],
                       'WF' : ['SHF']}

    test_onsets_df['orig_cond'] =  test_onsets_df['condition']

    for index, row in test_onsets_df.iterrows():
        if row['condition'] in test_labels['WP']:
            test_onsets_df.loc[index, 'condition'] = 'WP'
        elif row['condition'] in test_labels['WF']:
            test_onsets_df.loc[index, 'condition']  = 'WF'
        else:
            # new items & no response
            test_onsets_df.loc[index, 'condition']  = 'OTHER'

    return study_onsets_df, test_onsets_df


if __name__ == "__main__":
    main()

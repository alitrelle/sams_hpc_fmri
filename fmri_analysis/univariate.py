# code written by S Guerin and updated by A Trelle #

from glob import glob
import os.path
from os.path import exists
from mvpa_functions_Dec2018 import *
from itertools import chain


def main():
    subs2run = list(pd.read_csv('subs_paper1_final.csv').iloc[:,1])

    for subject in subs2run:

        # Handle files and load BOLD data
        # ----------------------------------------------------------------------
        rois = ['VTC','ANG','HC','CADG_ant','CA3DG','CA1','Sub','CADG_post']

        files = manage_files(subject)


        bold_df_test = pd.DataFrame(columns = ['Run','Onset','Condition','Word',\
                    'ANG','VTC','PHC','PRC','ERC','HC','CADG_ant','CA3DG','CA1','Sub','CADG_post'])
        bold_df_study = pd.DataFrame(columns = ['Run','Onset','Condition','Word',\
                    'ANG','VTC','PHC','PRC','ERC','HC','CADG_ant','CA3DG','CA1','Sub','CADG_post'])

        for region in rois:

            roi_name = region
            if region == 'VTC':
                roi_file = '/scratch/users/atrelle/AM/data/{}/masks/VTC_fusi_para_inftemp.nii.gz'.format(subject)
            elif region == 'ANG':
                roi_file = '/scratch/users/atrelle/AM/data/{}/masks/YeoANG.nii.gz'.format(subject)
            else:
                roi_file = '/scratch/users/atrelle/AM/data/{}/masks/{}.nii.gz'.format(subject,region)

            if exists(roi_file):

                # We want to load the study and test data together in one matrix (and
                # then split) to ensure that masking is equivalent for the two (out of
                # brain voxels are excluded by load_data by default)
                print('Loading bold data from {} for subject {}'.format(region,subject))

                big_X, big_X_groups, original_mask_cols = load_data(files['bold_files'],roi_file)

                # Estimate regression modeling effects of head motion and artifacts on
                # fMRI time series and get residuals
                # ----------------------------------------------------------------------
                DM = get_dm(big_X_groups, motion_files=files['motion_files'],
                        artifact_files=files['artifact_files'])
                betas, big_X_resid = glm(DM, big_X)

                # Split up the data into study and test components
                big_X_study, big_X_test, big_X_groups_study, big_X_groups_test = \
                    split_data(big_X_resid, big_X_groups, files)

                # Concat onsets and drop trials with artifacts
                # ----------------------------------------------------------------------
                study_onsets_df, test_onsets_df = get_onsets(files)

                # Average peak for each trial
                # ----------------------------------------------------------------------
                X_study_all_trials = avg_peak(big_X_study, study_onsets_df, peak_TRs=[2, 3, 4])
                X_test_all_trials = avg_peak(big_X_test, test_onsets_df,
                                             peak_TRs=[3, 4, 5])

                bold_df_study[region] = np.nanmean(X_study_all_trials,axis=1)
                bold_df_test[region] = np.nanmean(X_test_all_trials,axis=1)


        # Add other fields to study and test df:
        bold_df_test['Run'] = np.array(test_onsets_df['run'])
        bold_df_test['Onset'] = np.array(test_onsets_df['concat_onset'])
        bold_df_test['Condition'] = np.array(test_onsets_df['condition'])
        bold_df_test['Word'] = test_onsets_df['word']

        bold_df_study['Run'] = np.array(study_onsets_df['run'])
        bold_df_study['Onset'] = study_onsets_df['concat_onset']
        bold_df_study['Condition'] = np.array(study_onsets_df['condition'])
        bold_df_study['Word'] = study_onsets_df['word']

        bold_df_study.to_csv('./trialwise_activity/trialwise_activity_study_{}.csv'.\
            format(subject))
        bold_df_test.to_csv('./trialwise_activity/trialwise_activity_test_{}.csv'.\
            format(subject))

def manage_files(subject):

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

    files = {'bold_files' : bold_files,
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

def get_onsets(files):
    # Handle onsets
    study_onsets_ALL_df = concat_onsets(files['study_onsets_file'], \
        files['study_bold_files'])
    study_onsets_df = remove_artifact_trials(study_onsets_ALL_df, \
        files['study_artifact_files'])

    test_onsets_ALL_df = concat_onsets(files['test_onsets_file'], \
        files['test_bold_files'])
    test_onsets_df = remove_artifact_trials(test_onsets_ALL_df, \
        files['test_artifact_files'])

    return study_onsets_df, test_onsets_df

def split_data(big_X_resid, big_X_groups, files):
    n_study_runs = files['n_study_runs']
    n_test_runs = files['n_test_runs']

    # Split data into study and test, use residuals from regression model
    big_X_study = big_X_resid[big_X_groups <= n_study_runs, :]
    big_X_test = big_X_resid[big_X_groups > n_study_runs, :]

    big_X_groups_study = big_X_groups[big_X_groups <= n_study_runs]
    big_X_groups_test = big_X_groups[big_X_groups > n_study_runs]

    return big_X_study, big_X_test, big_X_groups_study, big_X_groups_test

if __name__ == "__main__":
    main()

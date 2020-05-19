# code written by S Guerin and updated by A Trelle #

from glob import glob
from os.path import exists
from mvpa_functions import *

def main():
    subs2run = list(pd.read_csv('subs_paper1_final.csv').iloc[:,1])
    standardize_across_voxels=False
    standardize_across_trials=True

    for subject in subs2run:

        # Handle files and load BOLD data
        # ----------------------------------------------------------------------
        roi_name = 'ANG'
        roi_file = '/scratch/users/atrelle/AM/data/{}/masks/YeoANG.nii.gz'.format(subject)
        output_file = './paper_one/mvpa_encoding/enc_cross_{}_{}_prob.csv'.format(subject,roi_name)

        if exists(roi_file):
            if not exists(output_file):
                files = manage_files(subject,roi_file)

                # Load study data, check mask for out of brain voxels
                print('Loading bold data for subject {}'.format(subject))
                big_X, big_X_groups, original_mask_cols = load_data(files['study_bold_files'],
                                                                    files['roi_file'])

                # Estimate regression modeling effects of head motion and artifacts on
                # fMRI time series and get residuals
                # ----------------------------------------------------------------------
                DM = get_dm(big_X_groups, motion_files=files['motion_files'],
                        artifact_files=files['artifact_files'])

                betas, big_X_resid = glm(DM, big_X)

                # Get category labels for machine learning classification (drops trials
                # with artifacts from onsets)
                # ----------------------------------------------------------------------
                onsets_df = get_onsets(files)

                # Average peak for each trial (optional z-scoring) Removes trials with artifacts
                # ----------------------------------------------------------------------
                X = avg_peak(big_X_resid, onsets_df, peak_TRs=[2, 3, 4])

                y=onsets_df.condition
                groups=onsets_df.run
                words=onsets_df.word

                # get uni from selected voxels
                y_study = get_binary_y(y)

                voxels2use,cat0_voxels,cat1_voxels,cat0_tvals, cat1_tvals = \
                    balanced_feat_sel(X, y_study, tot_num_vox=500)

                X_study_face = X[:,cat0_voxels].mean(axis=1)
                X_study_place = X[:,cat1_voxels].mean(axis=1)
                X_study_all = X[:,voxels2use].mean(axis=1)
                X_study_orig = X.mean(axis=1)

                if standardize_across_voxels==True:
                    # z-score across voxels
                    X = zscore(X,axis=1)

                if standardize_across_trials==True:
                    for run in pd.unique(onsets_df['run']):
                        # z-score across trials (within run)
                        X[(groups==run),:] = zscore(X[(groups==run),:],axis=0)

                # Run LogReg w/ Cross-Validation for encoding
                print('Encoding cross-validation for subject {}'.format(subject))
                auc_df,prob_df = CV_LogReg_Permutation(X, y, groups, onsets_df,
                    tot_num_vox=500,n_permutations=0,subsample_iterations=10)

                prob_df['Word'] = words
                prob_df['Onset']=onsets_df['concat_onset']
                prob_df.loc[:,'Face_Uni'] = X_study_face
                prob_df.loc[:,'Place_Uni'] = X_study_place
                prob_df.loc[:,'Sel_Uni'] = X_study_all
                prob_df.loc[:,'Uni'] = X_study_orig

                # Save output files
                auc_df.to_csv('./paper_one/mvpa_encoding/enc_cross_{}_{}_auc.csv'.format(subject,roi_name))
                prob_df.to_csv('./paper_one/mvpa_encoding/enc_cross_{}_{}_prob_uni.csv'.format(subject,roi_name))

    return None

def manage_files(subject,roi_file):

    # Get paths to files we need
    study_bold_files = sorted(glob(('/scratch/users/atrelle/AM/analysis/study/{}/reg/'
        + 'epi/unsmoothed/run_*/timeseries_xfm.nii.gz').format(subject)))

    study_onsets_file = \
        '/scratch/users/atrelle/AM/data/{}/behav/study_rsa_RT.csv'.format(subject)

    # Get motion & artifact files for all study and test runs (glob)
    motion_files = sorted(glob(('/scratch/users/atrelle/AM/analysis/study/{}/preproc/run_*/'
        + 'realignment_params.csv').format(subject)))

    artifact_files = sorted(glob(('/scratch/users/atrelle/AM/analysis/study/{}/preproc/'
        + 'run_*/artifacts.csv').format(subject)))

    n_study_runs = len(study_bold_files)

    files = {'roi_file' : roi_file,
             'study_bold_files' : study_bold_files,
             'study_onsets_file' : study_onsets_file,
             'motion_files' : motion_files,
             'artifact_files' : artifact_files,
             'n_study_runs' : n_study_runs}

    return files

def get_onsets(files):
    # Handle onsets
    study_onsets_ALL_df = concat_onsets(files['study_onsets_file'], \
        files['study_bold_files'])
    onsets_df = remove_artifact_trials(study_onsets_ALL_df, \
        files['artifact_files'])



    return onsets_df


if __name__ == "__main__":
    main()

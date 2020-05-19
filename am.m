% Wrapper script for Associative Memory (am) task in older adults
% This script loads the paths for the experiment, creates a stimulus order
% file for a given subject, and runs the task
%
% written by VC and SG 08/19/13, adapted from BK's RIFS task
% github repo: https://github.com/sgagnon/Experiments
%

% define and load paths
[S,thePath] = setupScript();

% (hard coded) define if stim presentation is alternating S/T, or blocked
presOrder = 'a'; % 'a' for alternating, 'b' for blocked

% define subject-specific info
subID = input('What is the subject ID? ','s');

% determine whether stims need to be assigned and order lists created
cd(thePath.orderfiles);

% check_stim added by SAG 1/12/2017:
check_stim = input('Is this the first study session in the PRACTICE SESSION? (1=yes, 0=no) ');
if check_stim==1
    useStims = 0;
elseif check_stim==0
    useStims = 1;
end

cd(thePath.scripts);
if useStims == 0 % create order lists, then run task
    am_stimAssigner(subID,thePath);
    if strcmp (presOrder,'a')
        am_makeOrder(subID,thePath);
    elseif strcmp (presOrder,'b')
        am_makeOrderB(subID,thePath);
    end
    am_prac_stimAssigner(subID,thePath);
    am_prac_makeOrder(subID,thePath);
    
    am_run(subID,thePath);
elseif useStims == 1 % run task
    am_run(subID,thePath);
end

clear all;
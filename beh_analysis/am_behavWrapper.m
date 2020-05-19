% Code for creating text files reflecting trial-specific data for each 
% study and test block of the AM task, as well as a summary file with 
% performance information
%
% vc 09/04/13

function am_behavWrapper
%% Set Paths

% define and load paths
currDir = pwd;
cd ..
[S,thePath] = setupScript();

% define subject-specific info
subID = input('What is the subject ID? ','s');

%% Output test files for study blocks, test blocks, and performance

% Study blocks
[studyData] = am_behavAnalysis_study(thePath,subID);

% Test blocks
[theData] = am_behavAnalysis_test(thePath,subID,studyData);

% Summary file with accuracy and RT
am_behavAnalysis_summary(thePath,subID,studyData,theData)

end

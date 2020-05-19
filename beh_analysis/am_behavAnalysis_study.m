function [studyData] = am_behavAnalysis_study(thePath,subID)
% Outputs trial information for each study run

% Load in mat file
cd(fullfile(thePath.data,subID));
studyMat = [subID '_study_cat.mat'];
load(studyMat);
nRuns = size(studyData,2);
studyData = studyData(nRuns); % create var inclusive of all info

% Create txt file
studyTxt = [subID '_behav_study.txt'];
fid = fopen(studyTxt,'wt');
fprintf(fid, 'index \t run \t trial \t onset \t duration \t cond \t word \t pic \t resp \t respRT \t ISIresp \t ISIrespRT \n');
formatString = '%d \t %d \t %d \t %.4f \t %.4f \t %s \t %s \t %s \t %s \t %.4f \t %s \t %.4f \n';

trialsPerRun = 24;
totalTrials = length(studyData.index);
for t = 1:totalTrials
    run = studyData.block(t);
    trial = t - trialsPerRun*(run - 1);
    onset = studyData.onset(t);
    dur = studyData.dur(t);
    cond = studyData.cond{t};
    word = studyData.wordShown{t};
    pic = studyData.picShown{t};
    resp = studyData.stimResp{t};
    respRT = studyData.stimRT{t};
    isiResp = studyData.isiResp{t};
    isiRespRT = studyData.isiRT{t};
    
    fprintf(fid, formatString, t, run, trial, onset, dur, cond, word, pic, resp, respRT,isiResp, isiRespRT);
end

cd(thePath.scripts);

end
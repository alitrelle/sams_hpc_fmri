function [theData] = am_behavAnalysis_test(thePath,subID,studyData)
% Outputs trial information for each test run

% Load in mat file
currDir = pwd;
cd(fullfile(thePath.data,subID));
testMat = [subID '_block5_test.mat']; % use this instead of concat bc it has resp mapping
load(testMat);

% Determine accuracy
trialsPerRun = 30;
totalTrials = length(theData.index);

for t = 1:totalTrials
    
    % convert button box resp during stim presentation to cond code
    % (if made more than one resp, take *LAST* resp)
    resp = theData.stimResp{t};
    if strcmp(resp,'noanswer')
        resp = resp;
    else
        nResps = length(resp);
        resp = resp(nResps - 1);
    end
    
    if leadIn == 10.2 %inside scanner using button box
        if strcmp(resp,'2')
            theData.stimCodedResp{t} = respCodes{1};
        elseif strcmp(resp,'3')
            theData.stimCodedResp{t} = respCodes{2};
        elseif strcmp(resp,'4')
            theData.stimCodedResp{t} = respCodes{3};
        elseif strcmp(resp,'6')
            theData.stimCodedResp{t} = respCodes{4};
        else
            theData.stimCodedResp{t} = 'NR';
        end
    elseif leadIn == 1 %outside scanner using laptop keyboard
        if strcmp(resp,'1')
            theData.stimCodedResp{t} = respCodes{1};
        elseif strcmp(resp,'2')
            theData.stimCodedResp{t} = respCodes{2};
        elseif strcmp(resp,'3')
            theData.stimCodedResp{t} = respCodes{3};
        elseif strcmp(resp,'4')
            theData.stimCodedResp{t} = respCodes{4};
        else
            theData.stimCodedResp{t} = 'NR';
        end
    end
        
    
    % if made more than one resp during stim presentation, take *LAST* RT
    trialRT = theData.stimRT{t};
    nRTs = length(trialRT);
    theData.stimRT{t} = trialRT(nRTs);
    
    % convert button box resp during ISI to cond code
    % (if made more than one resp, take *FIRST* resp)
    isiResp = theData.isiResp{t};
    if strcmp(isiResp,'noanswer')
        isiResp = isiResp;
    else
        nIsiResps = length(isiResp);
        isiResp = isiResp(nIsiResps - 1);
    end
    
    if strcmp(isiResp,'2')
        theData.isiCodedResp{t} = respCodes{1};
    elseif strcmp(isiResp,'3')
        theData.isiCodedResp{t} = respCodes{2};
    elseif strcmp(isiResp,'4')
        theData.isiCodedResp{t} = respCodes{3};
    elseif strcmp(isiResp,'6')
        theData.isiCodedResp{t} = respCodes{4};
    else
        theData.isiCodedResp{t} = 'NR';
    end
    
    % if made more than one resp during ISI, take *FIRST* RT
    trialIsiRT = theData.isiRT{t};
    theData.isiRT{t} = trialIsiRT(1);
    
    % depending on resp, determine accuracy
    if strcmp(theData.cond(t), 'F') % foil trials
        
        % responses during stim
        if strcmp(theData.stimCodedResp(t),'F')
            theData.acc{t} = 'CR'; % correct rejection
            theData.accSpec{t} = 'CR'; % correct rejection
        elseif strcmp(theData.stimCodedResp(t),'T')
            theData.acc{t} = 'FA'; % false alarm
            theData.accSpec{t} = 'FAI'; % false alarm, item
        elseif strcmp(theData.stimCodedResp(t),'TF')
            theData.acc{t} = 'FA'; % false alarm
            theData.accSpec{t} = 'FAF'; % false alarm, face
        elseif strcmp(theData.stimCodedResp(t),'TP')
            theData.acc{t} = 'FA'; % false alarm
            theData.accSpec{t} = 'FAP'; % false alarm, place
        else
            theData.acc{t} = 'NR'; % no response
            theData.accSpec{t} = 'NR'; % no response
        end
        
        % responses during ISI
        if strcmp(theData.isiCodedResp(t),'F')
            theData.accISI{t} = 'CR'; % correct rejection
            theData.accSpecISI{t} = 'CR'; % correct rejection
        elseif strcmp(theData.isiCodedResp(t),'T')
            theData.accISI{t} = 'FA'; % false alarm
            theData.accSpecISI{t} = 'FAI'; % false alarm
        elseif strcmp(theData.isiCodedResp(t),'TF')
            theData.accISI{t} = 'FA'; % false alarm
            theData.accSpecISI{t} = 'FAF'; % false alarm, face
        elseif strcmp(theData.isiCodedResp(t),'TP')
            theData.accISI{t} = 'FA'; % false alarm
            theData.accSpecISI{t} = 'FAP'; % false alarm, place
        else
            theData.accISI{t} = 'NR'; % no response
            theData.accSpecISI{t} = 'NR'; % no response
        end
        
    elseif strcmp(theData.cond(t), 'TF') % target face trials
        
        % responses during stim
        if strcmp(theData.stimCodedResp(t),'F')
            theData.acc{t} = 'M'; % miss
            theData.accSpec{t} = 'MF'; % miss, face trial
        elseif strcmp(theData.stimCodedResp(t),'T')
            theData.acc{t} = 'I'; % item only
            theData.accSpec{t} = 'IF'; % item only, face trial
        elseif strcmp(theData.stimCodedResp(t),'TF')
            theData.acc{t} = 'SH'; % source hit
            theData.accSpec{t} = 'SHF'; % source hit, face trial
        elseif strcmp(theData.stimCodedResp(t),'TP')
            theData.acc{t} = 'SM'; % source miss
            theData.accSpec{t} = 'SMF'; % source miss, face trial
        else
            theData.acc{t} = 'NR'; % no response
            theData.accSpec{t} = 'NR'; % no response
        end
        
        % responses during ISI
        if strcmp(theData.isiCodedResp(t),'F')
            theData.accISI{t} = 'M'; % miss
            theData.accSpecISI{t} = 'MF'; % miss, face trial
        elseif strcmp(theData.isiCodedResp(t),'T')
            theData.accISI{t} = 'I'; % item only
            theData.accSpecISI{t} = 'IF'; % item only, face trial
        elseif strcmp(theData.isiCodedResp(t),'TF')
            theData.accISI{t} = 'SH'; % source hit
            theData.accSpecISI{t} = 'SHF'; % source hit, face trial
        elseif strcmp(theData.isiCodedResp(t),'TP')
            theData.accISI{t} = 'SM'; % source miss
            theData.accSpecISI{t} = 'SMF'; % source miss, face trial
        else
            theData.accISI{t} = 'NR'; % no response
            theData.accSpecISI{t} = 'NR'; % no response
        end
        
    elseif strcmp(theData.cond(t), 'TP') % target place trials
        
        % responses during stim
        if strcmp(theData.stimCodedResp(t),'F')
            theData.acc{t} = 'M'; % miss
            theData.accSpec{t} = 'MP'; % miss, place trial
        elseif strcmp(theData.stimCodedResp(t),'T')
            theData.acc{t} = 'I'; % item only
            theData.accSpec{t} = 'IP'; % item only, place trial
        elseif strcmp(theData.stimCodedResp(t),'TF')
            theData.acc{t} = 'SM'; % source miss
            theData.accSpec{t} = 'SMP'; % source miss, place trial
        elseif strcmp(theData.stimCodedResp(t),'TP')
            theData.acc{t} = 'SH'; % source hit
            theData.accSpec{t} = 'SHP'; % source hit, place trial
        else
            theData.acc{t} = 'NR'; % no response
            theData.accSpec{t} = 'NR'; % no response
        end
        
        % responses during ISI
        if strcmp(theData.isiCodedResp(t),'F')
            theData.accISI{t} = 'M'; % miss
            theData.accSpecISI{t} = 'MP'; % miss, place trial
        elseif strcmp(theData.isiCodedResp(t),'T')
            theData.accISI{t} = 'I'; % item only
            theData.accSpecISI{t} = 'IP'; % item only, place trial
        elseif strcmp(theData.isiCodedResp(t),'TF')
            theData.accISI{t} = 'SM'; % source miss
            theData.accSpecISI{t} = 'SMP'; % source miss, place trial
        elseif strcmp(theData.isiCodedResp(t),'TP')
            theData.accISI{t} = 'SH'; % source hit
            theData.accSpecISI{t} = 'SHP'; % source hit, place trial
        else
            theData.accISI{t} = 'NR'; % no response
            theData.accSpecISI{t} = 'NR'; % no response
        end
        
    end
    
end

% Save new mat file with accuracy info
save([subID '_test_cat_acc.mat'],'theData');

% Create txt file
testTxt = [subID '_behav_test.txt'];
fid = fopen(testTxt,'wt');
fprintf(fid, ['index \t run \t trial \t onset \t duration \t cond \t target \t',...
    'associate \t resp \t acc \t accSpec \t respRT \t ISIresp \t ISIacc \t',...
    'ISIaccSpec \t ISIrespRT \n']);
formatString = ['%d \t %d \t %d \t %.4f \t %.4f \t %s \t %s \t ',...
    '%s \t %s \t %s \t %s \t %.4f \t %s \t %s \t',...
    '%s \t %.4f \n'];

trialsPerRun = 30;
totalTrials = length(theData.index);
for t = 1:totalTrials
    run = theData.block(t);
    trial = t - trialsPerRun*(run - 1);
    onset = theData.onset(t);
    dur = theData.dur(t);
    cond = theData.cond{t};
    target = theData.wordShown{t};
    % determine associate
    if strcmp(cond,'F')
        associate = 'foil';
    else
        aIdx = find(strcmp(target,studyData.wordShown));
        associate = studyData.picShown{aIdx};
    end
    resp = theData.stimCodedResp{t};
    acc = theData.acc{t};
    accSpec = theData.accSpec{t};
    respRT = theData.stimRT{t};
    isiResp = theData.isiCodedResp{t};
    isiAcc = theData.accISI{t};
    isiAccSpec = theData.accSpecISI{t};
    isiRespRT = theData.isiRT{t};
    
    fprintf(fid, formatString, t, run, trial, onset, dur, cond, target,...
        associate, resp, acc, accSpec, respRT, isiResp, isiAcc,...
        isiAccSpec, isiRespRT);
    
end

cd(currDir);

end
function am_makeOrderB(subID,thePath)

%% About this script
% This script will make the condition ordering for the AM task and assign the
% specific stims as well.  It will output the final lists and the only
% input it draws from is the list of the  stims that is specific to
% each subject and is created from a different script
% 
% Note: this version of the script differs from am_makeOrder, which creates
% stim ordering for alternating study/test blocks; in this script, all
% study blocks will be presented, followed by all test blocks, so stim
% ordering across test blocks can be randomized

%% initialize rand.
rand('twister',sum(100*clock));

%% Create the list composition: # of blocks & items/conditions per block

snum = subID;

% how many blocks are there
blockinfo.study.num = 5;
blockinfo.test.num = 5;

% how many trials per cond per block
blockinfo.study.facepair = 12;
blockinfo.study.placepair = 12;
blockinfo.test.facecue = 12;
blockinfo.test.placecue = 12;
blockinfo.test.foil = 6;

% items per cell
itemsPerCell.study = (blockinfo.study.facepair + blockinfo.study.placepair)*blockinfo.study.num;
itemsPerCell.test = (blockinfo.test.facecue + blockinfo.test.placecue + blockinfo.test.foil)*blockinfo.test.num;

% what are the conditions (using labels) that appear in each study block
conditions.study.block1 = {'WF' 'WP'};
conditions.study.block2 = {'WF' 'WP'};
conditions.study.block3 = {'WF' 'WP'};
conditions.study.block4 = {'WF' 'WP'};
conditions.study.block5 = {'WF' 'WP'};

% compute  total trials per study block
for a = 1:blockinfo.study.num
    blockinfo.study.size(a) = blockinfo.study.facepair + blockinfo.study.placepair;
    clear a;
end

% what are the conditions (using labels) that appear in each test block
conditions.test.block1 = {'TF' 'TP' 'F'};
conditions.test.block2 = {'TF' 'TP' 'F'};
conditions.test.block3 = {'TF' 'TP' 'F'};
conditions.test.block4 = {'TF' 'TP' 'F'};
conditions.test.block5 = {'TF' 'TP' 'F'};

% compute  total trials per test block
for a = 1:blockinfo.test.num
    blockinfo.test.size(a) = blockinfo.test.facecue + blockinfo.test.placecue + blockinfo.test.foil;
    clear a;
end

%% create condition ordering for study blocks

% Makes a vector for each block with the appropriate number of each
% condition label and then shuffles that vector

for c = 1:blockinfo.study.num
    TooManyRepeats = .5;
    while TooManyRepeats > 0;
        tempcon = eval(['conditions.study.block' num2str(c)]);
        g = ['block' num2str(c) '.study.conditions'];
        eval([g ' = [];']);
        for d = 1:blockinfo.study.facepair % (same trials per cond for either cond)
            eval([g ' = [' g ' tempcon];'])
        end
        eval([g ' = Shuffle(' g ');']);
%         eval([g ' = shuffle(' g ');']);
        eval(['f = length(' g ');']);
        % make a redundant variable that represents the current block (post
        % shuffle) so that it's easier to refer to it
        h = eval([g]);
        repcount = 0;
        for e = 1:f
            currcond = eval(['block' num2str(c) '.study.conditions{' num2str(e) '}']);
            eval(['block' num2str(c) '.study.term{' num2str(e) '} = h{e};']);
            currterm = eval(['block' num2str(c) '.study.term{' num2str(e) '}']);
            if strcmp(currcond(2),'F')
                eval(['block' num2str(c) '.study.ForS(' num2str(e) ') = 1;']);
            elseif strcmp(currcond(2),'P')
                eval(['block' num2str(c) '.study.ForS(' num2str(e) ') = 2;']);
            end
            
            if e > 1
                eval(['tmpj = block' num2str(c) '.study.ForS(' num2str(e) ');']);
                eval(['tmpk = block' num2str(c) '.study.ForS(' num2str(e-1) ');']);
                j=tmpj; % stupid hack for dealing with lack of j/k in the workspace
                k=tmpk;
                if j == k
                    repcount = repcount + 1;
                    eval(['block' num2str(c) '.study.repcount(' num2str(e) ') = ' num2str(repcount) ';']);
                else
                    repcount = 0;
                    eval(['block' num2str(c) '.study.repcount(' num2str(e) ') = ' num2str(repcount) ';']);
                end
            end
            
        end
          
        % the below number can be changed to limit the # of Face or Scene
        % Repeats that are allowed.  M = 2 means there can only be 3 Faces
        % or Scenes in a Row
        
        eval(['m = max(block' num2str(c) '.study.repcount);']);
        
        if m > 2
            TooManyRepeats = 1;
        else
            TooManyRepeats = 0;
        end
    end
    clear c d e f g h j k m currterm currcond tempcon;
end

%% create condition ordering for test blocks

% Makes a vector for each block with the appropriate number of each
% condition label and then shuffles that vector

for c = 1:blockinfo.study.num
    TooManyRepeats = .5;
    while TooManyRepeats > 0;
        g = ['block' num2str(c) '.test.conditions'];
        eval([g ' = [];']); 
        % first create vector of targets
        targs = eval(['conditions.test.block' num2str(c)]);
        targs = targs(1:2);
        for d = 1:blockinfo.test.facecue % (same trials per cond for either cond)
            eval([g ' = [' g ' targs];'])
        end
        % then add foils
        foils = eval(['conditions.test.block' num2str(c)]);
        foils = foils(3);
        startTrial = length(block1.test.conditions) + 1;
        endTrial = startTrial + blockinfo.test.foil - 1;
        for d = startTrial:endTrial
            eval([g ' = [' g ' foils];'])
        end
        % shuffle order
        eval([g ' = Shuffle(' g ');']);
        eval(['f = length(' g ');']);
        % make a redundant variable that represents the current block (post
        % shuffle) so that it's easier to refer to it
        h = eval([g]);
        repcount = 0;
        for e = 1:f
            currcond = eval(['block' num2str(c) '.test.conditions{' num2str(e) '}']);
            eval(['block' num2str(c) '.test.term{' num2str(e) '} = h{e};']);
            currterm = eval(['block' num2str(c) '.test.term{' num2str(e) '}']);
            if strcmp(currcond,'TF')
                eval(['block' num2str(c) '.test.ForS(' num2str(e) ') = 1;']);
            elseif strcmp(currcond,'TP')
                eval(['block' num2str(c) '.test.ForS(' num2str(e) ') = 2;']);
            elseif strcmp(currcond,'F')
                eval(['block' num2str(c) '.test.ForS(' num2str(e) ') = 3;']);
            end
            
            if e > 1
                eval(['tmpj = block' num2str(c) '.test.ForS(' num2str(e) ');']);
                eval(['tmpk = block' num2str(c) '.test.ForS(' num2str(e-1) ');']);
                j=tmpj; % stupid hack for dealing with lack of j/k in the workspace
                k=tmpk;
                if j == k
                    repcount = repcount + 1;
                    eval(['block' num2str(c) '.test.repcount(' num2str(e) ') = ' num2str(repcount) ';']);
                else
                    repcount = 0;
                    eval(['block' num2str(c) '.test.repcount(' num2str(e) ') = ' num2str(repcount) ';']);
                end
            end
            
        end
          
        % the below number can be changed to limit the # of Face or Scene
        % Repeats that are allowed.  M = 2 means there can only be 3 Faces
        % or Scenes in a Row
        
        eval(['m = max(block' num2str(c) '.test.repcount);']);
        
        if m > 2
            TooManyRepeats = 1;
        else
            TooManyRepeats = 0;
        end
    end
    clear c d e f g h j k m currterm currcond tempcon;
end

%% Make a single list of trials for both study and test

% Study
TrialCount = 0;
for a = 1:blockinfo.study.num
    eval(['currblock = block' num2str(a) '.study;']);
    for b = 1:length(currblock.conditions)
        TrialCount = TrialCount + 1;
        theOrder.study.index(TrialCount) = TrialCount;
        theOrder.study.block(TrialCount) = a;
        theOrder.study.cond(TrialCount) = currblock.conditions(b);
        theOrder.study.term(TrialCount) = currblock.term(b);
        if currblock.ForS(b) == 1
            theOrder.study.ForS{TrialCount} = 'Face';
        elseif currblock.ForS(b) == 2
            theOrder.study.ForS{TrialCount} = 'Place';
        end
        theOrder.study.repcount(TrialCount) = currblock.repcount(b);
    end
end
clear a b;

% Test
TrialCount = 0;
for a = 1:blockinfo.study.num
    eval(['currblock = block' num2str(a) '.test;']);
    for b = 1:length(currblock.conditions)
        TrialCount = TrialCount + 1;
        theOrder.test.index(TrialCount) = TrialCount;
        theOrder.test.block(TrialCount) = a;
        theOrder.test.cond(TrialCount) = currblock.conditions(b);
        theOrder.test.term(TrialCount) = currblock.term(b);
        if currblock.ForS(b) == 1
            theOrder.test.ForS{TrialCount} = 'Face';
        elseif currblock.ForS(b) == 2
            theOrder.test.ForS{TrialCount} = 'Place';
        end
        theOrder.test.repcount(TrialCount) = currblock.repcount(b);
    end
end
clear a b;

%% Study list: Assign specific stimuli to each condition

% load up the stims that were made by am_stimAssigner
subDir = fullfile(thePath.orderfiles, [subID]);
cd(subDir);
eval(['load ' subID '_stims.mat']);

% The below makes a list of the condition ID numbers for each condition
% that is shuffled within block 
allcons = {'WF' 'WP'};
nTrialsBlock = 12;
nBlocks = 5;
for p = 1:length(allcons)
    currcon = allcons{p};
    eval([currcon '_IDs = [];']);
    for q = 1:nBlocks 
        eval([currcon '_IDs = [' currcon '_IDs Shuffle( ((q-1)*' num2str(nTrialsBlock) ' +1) : q* ' num2str(nTrialsBlock) ')];']);
    end
    clear currcon;
end

% below is a counter for each condition that keeps track of what the next
% item from each condition number list should be.  NOTE: when 
% wordFace_counter is at 5, it doesn't mean that it will pull 
% wordFace_condID(5); this is because for each condition, the ID numbers 
% are shuffled (within block), so when wordFace_counter is at 5, it will 
% pull the 5th number from this shuffled vector.
WF_counter = 1;
WP_counter = 1;

% now assign the items
TotalTrialsStudy = sum(blockinfo.study.size);
for c = 1:TotalTrialsStudy
    currcon = theOrder.study.cond{c};
    currcounter = [currcon '_counter'];
    e = eval([currcounter]); % where we are in the counter
    d = eval([currcon '_IDs(e);']);
    theOrder.study.condID(c) = eval([currcon '.condID(' num2str(d) ');']);
    theOrder.study.wordID(c) = eval([currcon '.wordID(' num2str(d) ');']);
    theOrder.study.wordName(c) = eval([currcon '.wordName(' num2str(d) ');']);
    theOrder.study.imgID(c) = eval([currcon '.imgID(' num2str(d) ');']);
    theOrder.study.imgName(c) = eval([currcon '.imgName(' num2str(d) ');']);
    theOrder.study.imgFile(c) = eval([currcon '.imgFile(' num2str(d) ');']);
    theOrder.study.imgType(c) = eval([currcon '.imgType(' num2str(d) ');']);
    theOrder.study.subType(c) = eval([currcon '.subType(' num2str(d) ');']);
    theOrder.study.subsubType(c) = eval([currcon '.subsubType(' num2str(d) ');']);
    eval([currcounter ' = ' currcounter ' + 1;']);
    clear d e;
end

% now make mat and text files with the full list
cd(subDir);

clear a* S* c* p* q* r* Too* Trial*
% clear bl* F*

cmd = ['save ' subID '_studyList'];
eval(cmd);

txtName = [subID '_studyList.txt'];
fid = fopen(txtName, 'wt');
fprintf(fid, 'index\tblock\tcond\tterm\tForS\tcondID\twordID\twordName\timgID\timgName\timgFile\timgType\tsubType\tsubsubType\n');
for e = 1:TotalTrialsStudy
    fprintf(fid, '%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n',...
        theOrder.study.index(e), theOrder.study.block(e), theOrder.study.cond{e}, theOrder.study.term{e},...
        theOrder.study.ForS{e}, theOrder.study.condID{e}, theOrder.study.wordID{e}, theOrder.study.wordName{e}, ...
        theOrder.study.imgID{e}, theOrder.study.imgName{e}, theOrder.study.imgFile{e}, ...
        theOrder.study.imgType{e}, theOrder.study.subType{e}, theOrder.study.subsubType{e});
end

%% Create test lists for each block

% The below makes a list of the condition ID numbers for each condition
% that is shuffled across blocks
allcons = {'TF' 'TP' 'F'};
nTrialsBlock_bycon = {blockinfo.test.facecue, blockinfo.test.placecue, blockinfo.test.foil};
nBlocks = blockinfo.test.num;
for p = 1:length(allcons)
    currcon = allcons{p};
    nTrialsBlock = nTrialsBlock_bycon{p};
    totalTrials = nBlocks * nTrialsBlock;
    eval([currcon '_test_IDs = Shuffle(1:totalTrials);']);
    clear currcon;
end

% set counters
TF_counter = 1;
TP_counter = 1;
F_counter = 1;
studied_testcons = {'TF' 'TP'};
studied_studycons = {'WF' 'WP'};
foil_con = {'FOIL'};

% now assign the items
TotalTrialsTest = sum(blockinfo.test.size);
for c = 1:TotalTrialsTest

    currcon = theOrder.test.cond{c};
    currcounter = [currcon '_counter'];
    e = eval([currcounter]); % where we are in the counter
    d = eval([currcon '_test_IDs(e);']);

    % Foil
    if strcmp(currcon, 'F')
        theOrder.test.condID(c) = eval([currcon '.condID(' num2str(d) ');']);
        theOrder.test.wordID(c) = eval([currcon '.wordID(' num2str(d) ');']);
        theOrder.test.wordName(c) = eval([currcon '.wordName(' num2str(d) ');']);
        theOrder.test.ForS(c) = foil_con;
        theOrder.test.imgID(c) = foil_con;
        theOrder.test.imgName(c) = foil_con;
        theOrder.test.imgFile(c) = foil_con;
        theOrder.test.imgType(c) = foil_con;
        theOrder.test.subType(c) = foil_con;
        theOrder.test.subsubType(c) = foil_con;
        eval([currcounter ' = ' currcounter ' + 1;']);
        clear d e;

    % Studied
    else
        % Find corresponding study condition
        % Note: for this version of the task it's unnecessary to do this,
        % but since the code was already setup such that the stimuli are
        % saved in WP/WF structs, it's easier to pull from these using
        % shuffled TP/TF indices
        test_ind = strcmp(currcon, studied_testcons);
        study_con = studied_studycons{test_ind};

        % fill in info based on new shuffled IDs
        theOrder.test.condID(c) = eval([study_con '.condID(' num2str(d) ');']);
        theOrder.test.wordID(c) = eval([study_con '.wordID(' num2str(d) ');']);
        theOrder.test.wordName(c) = eval([study_con '.wordName(' num2str(d) ');']);
        theOrder.test.imgID(c) = eval([study_con '.imgID(' num2str(d) ');']);
        theOrder.test.imgName(c) = eval([study_con '.imgName(' num2str(d) ');']);
        theOrder.test.imgFile(c) = eval([study_con '.imgFile(' num2str(d) ');']);
        theOrder.test.imgType(c) = eval([study_con '.imgType(' num2str(d) ');']);
        theOrder.test.subType(c) = eval([study_con '.subType(' num2str(d) ');']);
        theOrder.test.subsubType(c) = eval([study_con '.subsubType(' num2str(d) ');']);
        eval([currcounter ' = ' currcounter ' + 1;']);
        clear d e;
    end      
end

% now make mat and text files with the full list
cd(subDir);

clear a* F* S* bl* c* p* q* r* Too* Trial*
cmd = ['save ' subID '_testList'];
eval(cmd);

txtName = [subID '_testList.txt'];
fid = fopen(txtName, 'wt');
fprintf(fid, 'index\tblock\tcond\tterm\tForS\tcondID\twordID\twordName\timgID\timgName\timgFile\timgType\tsubType\tsubsubType\n');
for e = 1:TotalTrialsTest
    fprintf(fid, '%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n',...
        theOrder.test.index(e), theOrder.test.block(e), theOrder.test.cond{e}, theOrder.test.term{e},...
        theOrder.test.ForS{e}, theOrder.test.condID{e}, theOrder.test.wordID{e}, theOrder.test.wordName{e}, ...
        theOrder.test.imgID{e}, theOrder.test.imgName{e}, theOrder.test.imgFile{e}, ...
        theOrder.test.imgType{e}, theOrder.test.subType{e}, theOrder.test.subsubType{e});
end



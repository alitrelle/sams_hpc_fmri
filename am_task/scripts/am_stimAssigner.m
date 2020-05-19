function am_stimAssigner(subID,thePath)

%% Makes the stim list for each subject
% This script calls up the basic list of all the stimuli that specifies the
% info for each word and img stimulus (kind, name, ID#, etc), and then
% creates a unique pairing of stimuli for each subject.  The ID of each
% INDIVIDUAL stimulus (e.g., Face#22) never changes across subjects (making item
% analyses easier), but once the stims are paired for each subject, there
% is a name that refers to that condition (e.g., Face-Scene # 15) that can
% be used for tracking an item WITHIN subject but has no meaning across
% subjects.  

%% initialize randomness  
rand('twister',sum(100*clock));

%% Some info about block & trial counts (for sub/subsubtype distribution)
% how many blocks are there
blockinfo.study.num = 5;

% how many trials per cond per block
blockinfo.study.facepair = 12;
blockinfo.study.placepair = 12;

% how to counterbalance within conditions
face_by_subtype = 1;


%% load up the stim list

[NUMERIC,TXT,RAW] = xlsread('inputlist.xls');

% Below things are grouped according to what should be shuffled together

% words (rows 2-151)
wordID = TXT(2:151,1);
wordName = TXT(2:151,2);

% Faces (rows 2-61)
faceID = TXT(2:61,3);
faceName = TXT(2:61,4);
faceFile = TXT(2:61,5);
actOrSing = TXT(2:61,6);
gender = TXT(2:61,7);

% Scenes (rows 2-61)
sceneID = TXT(2:61,8);
sceneName = TXT(2:61,9);
sceneFile = TXT(2:61,10);
domOrInt = TXT(2:61,11);
creator = TXT(2:61,12);


%% Shuffle each of the sets

% the words
[wordID, wordName] = paraShuffle(wordID, wordName);

% the Faces
[faceID, faceName, faceFile, actOrSing, gender] = paraShuffle(faceID, faceName, faceFile, actOrSing, gender);

% the Scenes
[sceneID, sceneName, sceneFile, domOrInt, creator] = paraShuffle(sceneID, sceneName, sceneFile, domOrInt, creator);


%% Make sure Condition subtypes are equally dispersed into blocks
if face_by_subtype == 1
    face_subtype = gender;
    face_subsubtypes = 2;
    face_subsublabels = {'male', 'female'};

    % set up ordering based on number of subtypes
    trialmat(1, :) = 1:1:(blockinfo.study.num * blockinfo.study.facepair);
    trialmat(2,:) = mod((trialmat(1, :)-1), face_subsubtypes) + 1;

    % get current indices for each subtype (rows=diff levels, columns=ind into level)
    subsub_ind = [];
    for i=1:face_subsubtypes
        subsub_ind(i,:)= find(strcmp(face_subsublabels{i}, face_subtype));
    end
    
    % reorder indices based on subtype
    counters(1:face_subsubtypes) = 1;
    ind_reordered = [];
    for i=1:length(faceID)
        cond = trialmat(2,i);
        ind_reordered(i) = subsub_ind(cond,counters(cond));
        counters(cond) = counters(cond) + 1;
        clear cond;
    end

    % pull correct info from face structures
    for i=1:length(faceID)
        faceID_by_subtype(i) = faceID(ind_reordered(i));
        faceName_by_subtype(i) = faceName(ind_reordered(i));
        faceFile_by_subtype(i) = faceFile(ind_reordered(i));
        actOrSing_by_subtype(i) = actOrSing(ind_reordered(i));
        gender_by_subtype(i) = gender(ind_reordered(i));
    end
    
    clear trialmat face_subsubtypes face_subsublabels subsub_ind counters
end

%% Assign specific stims to conditions and create pairs
if face_by_subtype == 1
    for a = 1:length(faceID)
        WF.condID{a} = ['WF' num2str(a)];
        WF.wordID(a) = wordID(a);
        WF.wordName(a) = wordName(a);
        WF.imgID(a) = faceID_by_subtype(a);
        WF.imgName(a) = faceName_by_subtype(a);
        WF.imgFile(a) = faceFile_by_subtype(a);
        WF.imgType{a} = ['face'];
        WF.subType(a) = actOrSing_by_subtype(a);
        WF.subsubType(a) = gender_by_subtype(a);
        clear a;
    end
else
    for a = 1:length(faceID)
        WF.condID{a} = ['WF' num2str(a)];
        WF.wordID(a) = wordID(a);
        WF.wordName(a) = wordName(a);
        WF.imgID(a) = faceID(a);
        WF.imgName(a) = faceName(a);
        WF.imgFile(a) = faceFile(a);
        WF.imgType{a} = ['face'];
        WF.subType(a) = actOrSing(a);
        WF.subsubType(a) = gender(a);
        clear a;
    end
end

for a = 1:60
    WP.condID{a} = ['WP_' num2str(a)];
    WP.wordID(a) = wordID(a+60);
    WP.wordName(a) = wordName(a+60);
    WP.imgID(a) = sceneID(a);
    WP.imgName(a) = sceneName(a);
    WP.imgFile(a) = sceneFile(a);
    WP.imgType{a} = ['place'];
    WP.subType(a) = domOrInt(a);
    WP.subsubType(a) = creator(a);
    clear a;
end
    
for a = 1:30
    F.condID{a} = ['F' num2str(a)];
    F.wordID(a) = wordID(a+120);
    F.wordName(a) = wordName(a+120);
    clear a;
end

cd(thePath.orderfiles);
subDir = fullfile(thePath.orderfiles, [subID]);
if ~exist(subDir)
   mkdir(subDir);
end
cd(subDir);
eval(['save ' subID '_stims WF WP F']);
cd(thePath.scripts);







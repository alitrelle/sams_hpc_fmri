function am_prac_stimAssigner(subID,thePath)
% borrowed from regular stim assigner

%% initialize randomness  
rand('twister',sum(100*clock));

% Setup num blocks + type counts
blockinfo.study.num = 1; %block nums
blockinfo.study.facepair = 5;
blockinfo.study.placepair = 5;

total_words = 13;
total_faces = 5;
total_places = 5;

% how to counterbalance within conditions
face_by_subtype = 0;

% stimlist
[NUMERIC,TXT,RAW] = xlsread('prac_inputlist.xls');


% Below things are grouped according to what should be shuffled together

% words (rows 2-#)
wordID = TXT(2:(total_words+1),1);
wordName = TXT(2:(total_words+1),2);

% Faces (rows 2-#)
faceID = TXT(2:(total_faces+1),3);
faceName = TXT(2:(total_faces+1),4);
faceFile = TXT(2:(total_faces+1),5);
actOrSing = TXT(2:(total_faces+1),6);
gender = TXT(2:(total_faces+1),7);

% Scenes (rows 2-#)
sceneID = TXT(2:(total_places+1),8);
sceneName = TXT(2:(total_places+1),9);
sceneFile = TXT(2:(total_places+1),10);
domOrInt = TXT(2:(total_places+1),11);
creator = TXT(2:(total_places+1),12);

%% Shuffle each of the sets
% the words
[wordID, wordName] = paraShuffle(wordID, wordName);

% the Faces
[faceID, faceName, faceFile, actOrSing, gender] = paraShuffle(faceID, faceName, faceFile, actOrSing, gender);

% the Scenes
[sceneID, sceneName, sceneFile, domOrInt, creator] = paraShuffle(sceneID, sceneName, sceneFile, domOrInt, creator);


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

for a = 1:length(sceneID)
    WP.condID{a} = ['WP_' num2str(a)];
    WP.wordID(a) = wordID(a+length(faceID));
    WP.wordName(a) = wordName(a+length(faceID));
    WP.imgID(a) = sceneID(a);
    WP.imgName(a) = sceneName(a);
    WP.imgFile(a) = sceneFile(a);
    WP.imgType{a} = ['place'];
    WP.subType(a) = domOrInt(a);
    WP.subsubType(a) = creator(a);
    clear a;
end
    
for a = 1:(length(wordID)-(length(faceID)+length(sceneID)))
    F.condID{a} = ['F' num2str(a)];
    F.wordID(a) = wordID(a+length(faceID)+length(sceneID));
    F.wordName(a) = wordName(a+length(faceID)+length(sceneID));
    clear a;
end

cd(thePath.orderfiles);
subDir = fullfile(thePath.orderfiles, [subID]);
if ~exist(subDir)
   mkdir(subDir);
end
cd(subDir);
eval(['save ' subID '_prac_stims WF WP F']);
cd(thePath.scripts);


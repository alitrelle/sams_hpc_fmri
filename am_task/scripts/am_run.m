
function am_run(subID,thePath)

taskType = input('S(1), T(2), PracS(3), PracT(4)?, FreeResp(5) ');

block = input('Enter block number: ');
if (taskType == 2) || (taskType == 4)
    respMap = input('Response mapping (1-4)? ');
end
S.scanner = input('inside scanner(1) or outside(2)? ');

S.study_name = 'AM';
S.subID = subID;

%% Set input device (keyboard or buttonbox)
if S.scanner == 1
    S.boxNum = getBoxNumber;  % buttonbox
    S.kbNum = getKeyboardNumberCurtis; % keyboard
elseif S.scanner == 2
    S.boxNum = BH1getKeyboardNumber;  % keyboard
    S.kbNum = BH1getKeyboardNumber; % keyboard
end

%% Set up subj-specific data directory
S.subData = fullfile(thePath.data, [subID]);
if ~exist(S.subData)
    mkdir(S.subData);
end
cd(S.subData);

%% Screen commands
S.screenNumber = 0;
S.screenColor = 0;
S.textColor = 0;
S.endtextColor = 255;
[S.Window, S.myRect] = Screen(S.screenNumber, 'OpenWindow', S.screenColor, [], 32);
Screen('TextSize', S.Window, 24);
Screen('TextStyle', S.Window, 1);
S.on = 1;  % Screen now on

%% Run Experiment Scripts
switch taskType
    
    % Run Study
    case 1
        listName = [subID '_studyList.txt'];
        saveName = [subID '_block' num2str(block) '_study'];
        cat_saveName = [subID '_study_cat']; % combines test data across blocks
        
        % Load in data from prev blocks
        try
            if block > 1
                load(cat_saveName);
                prevData = getPrevData(studyData, block);
            else
                prevData = struct;
            end
        catch err
            outputError(thePath.data, S.subData, err);
        end
        
        % Run study
        try
            studyData(block) = am_study(thePath,listName,S,block,saveName,prevData);
        catch err
            outputError(thePath.data, S.subData, err);
        end
        
        % Save out concatenated data
        save(cat_saveName, 'studyData');
        
        
        % Run Test
    case 2
        listName = [subID '_testList.txt'];
        saveName = [subID '_block' num2str(block) '_test'];
        cat_saveName = [subID '_test_cat']; % combines test data across blocks
        
        % Load in data from prev blocks
        try
            if block > 1
                load(cat_saveName);
                prevData = getPrevData(testData, block);
            else
                prevData = struct;
            end
        catch err
            outputError(thePath.data, S.subData, err);
        end
        
        % Run test
        try
            testData(block) = am_test(thePath,listName,S,respMap,block,saveName,prevData);
        catch err
            outputError(thePath.data, S.subData, err);
        end
        
        % Save out concatenated data
        save(cat_saveName, 'testData');
        
        % Save subj to group file
        if block == 1
            cd(thePath.data);
            dataFile =fopen([S.study_name,'_subids.txt'], 'a');
            fprintf(dataFile,([subID,'\n']));
        end
        
        % Run Practice Study
    case 3
        listName = [subID '_prac_studyList.txt'];
        saveName = [subID '_prac_study'];
        
        % Run practice
        try
            am_pracStudy(thePath,listName,S,block,saveName);
        catch err
            outputError(thePath.data, S.subData, err);
        end
        
        % Run Practice Test
    case 4
        listName = [subID '_prac_testList.txt'];
        saveName = [subID '_prac_test'];
        
        % Run practice
        try
            am_pracTest(thePath,listName,S,respMap,block,saveName);
        catch err
            outputError(thePath.data, S.subData, err);
        end
        
        % Run Post-test
    case 5
        cat_saveName = [subID '_freeresp_cat']; % combines test data across blocks
        
        % Load in the order file
        listName = [subID '_testList.txt'];
        subDir = fullfile(thePath.orderfiles, S.subID);
        cd(subDir);
        theList = read_table(listName);
        theData.index = theList.col1;
        theData.block = theList.col2;
        theData.cond = theList.col3;
        theData.term = theList.col4;
        theData.ForS = theList.col5;
        theData.condID = theList.col6;
        theData.wordID = theList.col7;
        theData.wordName = theList.col8;
        theData.imgID = theList.col9;
        theData.imgName = theList.col10;
        theData.imgFile = theList.col11;
        theData.imgType = theList.col12;
        theData.subType = theList.col13;
        theData.subsubType = theList.col14;
        listLength = length(theData.index);
        
        % Shuffle so post-test order is different than test order
        theDataOrig = theData;
        trialOrder = Shuffle([1:listLength]);
        dFields = fields(theData);
        nFields = length(dFields);
        for f = 3:nFields % don't shuffle the index or block fields
            field = dFields{f};
            theData.(field) = theData.(field)(trialOrder);
        end
        
        % Run 5 blocks of post-test
        for block_num = block:5
            saveName = [subID '_block' num2str(block_num) '_freeresp'];
            
            % Load in data from prev blocks
            try
                if block_num > 1
                    load(cat_saveName);
%                     prevData = getPrevData(freerespData, block_num);
                    theData = getPrevData(freerespData, block_num);
%                 else
%                     prevData = struct;
                end
            catch err
                outputError(thePath.data, S.subData, err);
            end
            
            % Run post-test
            try
                freerespData(block_num) = am_freeresp(thePath,S,block_num,saveName,theData);
            catch err
                outputError(thePath.data, S.subData, err);
            end
            
            % Save out concatenated data
            save(cat_saveName, 'freerespData');
            
            % Save subj to group file
            if block_num == 1
                cd(thePath.data);
                dataFile =fopen([S.study_name,'_subids.txt'], 'a');
                fprintf(dataFile,([subID,'\n']));
            end
        end
        
end

message = 'Press g to exit';
DrawFormattedText(S.Window,message,'center','center',S.endtextColor);
Screen(S.Window,'Flip');
pause;
clear screen;
cd(thePath.scripts);

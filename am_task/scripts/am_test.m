function theData = am_test(thePath,listName,S,respMap,block,saveName,theData)

%% initialize rand.
rand('twister',sum(100*clock));
kbNum=S.kbNum;

%% Come up with response labels
if respMap == 1
    respCodes = {'F' 'T' 'TF' 'TP'};
    labelReminder = 'response_mapping1.jpg';
elseif respMap == 2
    respCodes = {'F' 'T' 'TP' 'TF'};
    labelReminder = 'response_mapping2.jpg';
elseif respMap == 3
    respCodes = {'TF' 'TP' 'T' 'F'};
    labelReminder = 'response_mapping3.jpg';
elseif respMap == 4
    respCodes = {'TP' 'TF' 'T' 'F'};
    labelReminder = 'response_mapping4.jpg';
end


%% Read the input file
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


%% Trial Outline
stimTime = 4;
nullTime = 8;
% stimTime = 1;
% nullTime = .5;

if S.scanner == 1
    leadIn = 10;
    leadOut = 10;
elseif S.scanner == 2
    leadIn = 1;
    leadOut = 1;
end

%% Screen commands and device specification
screenColor = 0;
textColor = 255;
textColor2 = [0 255 0];

Window = S.Window;
myRect = S.myRect;
Screen(Window,'TextSize', 36);
Screen('TextFont', Window, 'Arial');
Screen('TextStyle', Window, 1);

% get center and box points
xcenter = myRect(3)/2;
ycenter = myRect(4)/2;

Screen(Window,'FillRect', screenColor);
Screen(Window,'Flip');

%% Remind subject about response options before starting

% Load reminder of key labels
cd(thePath.stim);
pic = imread(labelReminder);
labelRemind = Screen(Window,'MakeTexture', pic);

% Load blank
fileName = 'blank.jpg';
pic = imread(fileName);
blank = Screen(Window,'MakeTexture', pic);

% Print reminder
Screen(Window, 'DrawTexture', blank);
message = ['TEST BLOCK ' num2str(block) ' OF 5'];
DrawFormattedText(Window,message,'center',ycenter-400,textColor);
Screen(Window, 'DrawTexture', labelRemind);
Screen(Window,'Flip');
getKey('g',S.kbNum);

%% Pre-load images

% Load fixation
fileName = 'fix.jpg';
pic = imread(fileName);
fix = Screen(Window,'MakeTexture', pic);

% Load empty box
fileName = 'catChoice.jpg';
pic = imread(fileName);
[catChoiceheight catChoicewidth crap] = size(pic);
catChoicePtr = Screen(Window,'MakeTexture', pic);

%% Get everything else ready

% preallocate shit:
trialcount = 0;
for preall = 1:listLength
    if (theData.block(preall)==block)
        theData.onset(preall) = 0;
        theData.dur(preall) =  0;
        theData.stimResp{preall} = 'NR';
        theData.stimRT{preall} = 0;
        theData.stimCodedResp{preall} = 'NR';
        theData.isiResp{preall} = 'NR';
        theData.isiRT{preall} = 0;
        theData.isiCodedResp{preall} = 'NR';
        theData.wordShown{preall} = 'blank';
        theData.corrResp{preall} = 'NR';
    end
end

% get ready screen
Screen(Window, 'DrawTexture', blank);
message = ['Get ready!'];
DrawFormattedText(Window,message,'center','center',textColor);
Screen(Window,'Flip');

% get cursor out of the way
SetMouse(0,myRect(4));

% initiate experiment and begin recording time...
status = 1;
while 1
    getKey('g',S.kbNum);
    if S.scanner == 1
%         [status, startTime] = startScan;
        [status, startTime] = startScan_eprime;
    else
        status = 0;
        S.boxNum = S.kbNum;
        startTime = GetSecs;
    end
    if status == 0 % status=0 when startScan is successful
        break
    end
end

%% Start task

Priority(MaxPriority(Window));
goTime = 0;

% Show fixation (lead-in)
goTime = goTime + leadIn;
Screen(Window, 'DrawTexture', blank);
Screen(Window, 'DrawTexture', fix);
Screen(Window,'Flip');
recordKeys(startTime,goTime,kbNum);

cd(S.subData); % for saving data

% Loop through stimulus trials
for Trial = 1:listLength
    if (theData.block(Trial)==block)
        
        goTime = goTime + stimTime - 1; % will change text color for last sec of stimTime
%         goTime = goTime + stimTime;
        theData.onset(Trial) = GetSecs - startTime;
        
        keys = {'NR'};
        RT = 0;
        
        % Blank
        Screen(Window, 'DrawTexture', blank);
        
        % Display empty box prompting category choice
        destRect = [xcenter-catChoicewidth/2 ycenter-catChoiceheight/2 xcenter+catChoicewidth/2 ycenter+catChoiceheight/2];
        Screen('DrawTexture',Window,catChoicePtr,[],destRect);
        
        % Draw word in white during first 3 sec of stimulus presentatio
        word = theData.wordName{Trial}(3:end);
        DrawFormattedText(Window,word,'center',ycenter-220,textColor);
        
        theData.wordShown{Trial} = word;
        Screen(Window,'Flip');
        
        % Collect responses during first 3 sec of stimulus presentation
        [keys RT] = recordKeys(startTime,goTime,S.boxNum);
        
%         if S.scanner == 2
%             if strcmp(keys(end-1:end),'1!')
%                 codedResp = respCodes{1};
%             elseif strcmp(keys(end-1:end),'2@')
%                 codedResp = respCodes{2};
%             elseif strcmp(keys(end-1:end),'3#')
%                 codedResp = respCodes{3};
%             elseif strcmp(keys(end-1:end),'4$')
%                 codedResp = respCodes{4};
%             else
%                 codedResp = 'NR';
%             end
%         elseif S.scanner == 1
%             if strcmp(keys(end-1:end),'1!')
%                 codedResp = respCodes{1};
%             elseif strcmp(keys(end-1:end),'2@')
%                 codedResp = respCodes{2};
%             elseif strcmp(keys(end-1:end),'3#')
%                 codedResp = respCodes{3};
%             elseif strcmp(keys(end-1:end),'4$')
%                 codedResp = respCodes{4};
%             else
%                 codedResp = 'NR';
%             end
%         end
%          
%         if RT < 0.001 % so that RTs of 0 will stand out if they get averaged accidentally
%             RT = 999;
%         end
%         
%         theData.stimResp{Trial} = keys;
%         theData.stimRT{Trial} = RT;
%         theData.stimCodedResp{Trial} = codedResp;
%         theData.corrResp{Trial} = theData.cond{Trial};
        
        % Re-display empty box prompting category choice
        destRect = [xcenter-catChoicewidth/2 ycenter-catChoiceheight/2 xcenter+catChoicewidth/2 ycenter+catChoiceheight/2];
        Screen('DrawTexture',Window,catChoicePtr,[],destRect);        

        % Draw word in green during last sec of stimulus presentation
        goTime = goTime + 1;
        word = theData.wordName{Trial}(3:end);
        DrawFormattedText(Window,word,'center',ycenter-220,textColor2);
        Screen(Window,'Flip');
        
        % Collect responses during last sec of stimulus presentation
        [keys2 RT2] = recordKeys(startTime,goTime,S.boxNum);
        RT2 = RT2 + 3;
        if strcmp(keys,'noanswer') && ~strcmp(keys2,'noanswer')
            keys = keys2;
            RT = RT2;
        elseif ~strcmp(keys,'noanswer') && ~strcmp(keys2,'noanswer')
            keys = keys2;
            RT = RT2;
        end
        
        % Code responses
        if S.scanner == 2
            if length(keys) == 1 % in the off chance the subject hits a non-numeric button on the keyboard
                codedResp = keys;
            elseif strcmp(keys(end-1:end),'1!')
                codedResp = respCodes{1};
            elseif strcmp(keys(end-1:end),'2@')
                codedResp = respCodes{2};
            elseif strcmp(keys(end-1:end),'3#')
                codedResp = respCodes{3};
            elseif strcmp(keys(end-1:end),'4$')
                codedResp = respCodes{4};
            else
                codedResp = 'NR';
            end
        elseif S.scanner == 1
            if strcmp(keys(end-1:end),'1!')
                codedResp = respCodes{1};
            elseif strcmp(keys(end-1:end),'2@')
                codedResp = respCodes{2};
            elseif strcmp(keys(end-1:end),'3#')
                codedResp = respCodes{3};
            elseif strcmp(keys(end-1:end),'4$')
                codedResp = respCodes{4};
            else
                codedResp = 'NR';
            end
        end
         
        if RT < 0.001 % so that RTs of 0 will stand out if they get averaged accidentally
            RT = 999;
        end
        
        theData.stimResp{Trial} = keys;
        theData.stimRT{Trial} = RT;
        theData.stimCodedResp{Trial} = codedResp;
        theData.corrResp{Trial} = theData.cond{Trial};
        
        % Present fixation during ITI
        goTime = goTime + nullTime;
        Screen(Window, 'DrawTexture', blank);
        Screen(Window, 'DrawTexture', fix);
        Screen(Window,'Flip');
        
        % Collect responses during ITI
        [keys RT] = recordKeys(startTime,goTime,S.boxNum);
        
        % Code ITI responses
        if S.scanner == 2
            if length(keys) == 1 % in the off chance the subject hits a non-numeric button on the keyboard
                codedResp = keys;
            elseif strcmp(keys(end-1:end),'1!')
                codedResp = respCodes{1};
            elseif strcmp(keys(end-1:end),'2@')
                codedResp = respCodes{2};
            elseif strcmp(keys(end-1:end),'3#')
                codedResp = respCodes{3};
            elseif strcmp(keys(end-1:end),'4$')
                codedResp = respCodes{4};
            else
                codedResp = 'NR';
            end
        elseif S.scanner == 1
            codedResp = 'buttonbox';
        end
        
        if RT < 0.001 % so that RTs of 0 will stand out if they get averaged accidentally
            RT = 999;
        end
        
        theData.isiResp{Trial} = keys;
        theData.isiRT{Trial} = RT;
        theData.isiCodedResp{Trial} = codedResp;
        
        % Record trial duration
        theData.dur(Trial) = (GetSecs - startTime) - theData.onset(Trial); % duration from stim onset
        
        % Save to mat file
        matName = [saveName '.mat'];
        cmd = ['save ' matName];
        eval(cmd);
    end
end

% Show fixation (lead-out)
goTime = goTime + leadOut;
Screen(Window, 'DrawTexture', blank);
Screen(Window, 'DrawTexture', fix);
Screen(Window,'Flip');
recordKeys(startTime,goTime,kbNum);

Screen(Window, 'DrawTexture', blank);
Screen(Window,'Flip');

Priority(0);
end


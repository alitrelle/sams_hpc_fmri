function theData = am_freeresp(thePath,S,block,saveName,theData)
% written by steph for AP task; edited by vc for AM task
% specifically, edited to only test targets and not foils

%% initialize rand.
rand('twister',sum(100*clock));
kbNum=S.kbNum;


%% Set dir
subDir = fullfile(thePath.orderfiles, S.subID);
cd(subDir);

%% Trial Outline
stimTime = 4;
nullTime = 1;

leadIn = 2;
leadOut = 2;

%% Screen commands and device specification
S.textColor = 255; % was set to black in 'S' variable that was loaded in
S.textColorResp = 255;
fontsize_type = 18;
lengthLine = 40; % num characters in line

RETURN = 10;
DELETE = 8;

Window = S.Window;
myRect = S.myRect;
Screen(Window,'TextSize', 36);
Screen('TextFont', Window, 'Helvetica');
Screen('TextStyle', Window, 1);

% get center and box points
xcenter = myRect(3)/2;
ycenter = myRect(4)/2;
yCoor_type = ycenter+200;

Screen(Window,'FillRect', S.screenColor);
Screen(Window,'Flip');


%% Remind subject about response options before starting

% Load reminder of key labels
labelReminder = 'response_freeresp.jpg';
cd(thePath.stim);
pic = imread(labelReminder);
labelRemind = Screen(Window,'MakeTexture', pic);

% Load blank
fileName = 'blank.jpg';
pic = imread(fileName);
blank = Screen(Window,'MakeTexture', pic);

% Print reminder
Screen(Window,'FillRect', S.screenColor);
message = ['POST-TEST BLOCK ' num2str(block)];
DrawFormattedText(Window,message,'center',ycenter-400,S.textColor);
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
for preall = 1:length(theData.index)
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
Screen(Window,'FillRect', S.screenColor);
message = ['Get ready! (press g to begin)'];
DrawFormattedText(Window,message,'center','center',S.textColor);
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
Screen(Window,'FillRect', S.screenColor);
Screen(Window, 'DrawTexture', fix);
Screen(Window,'Flip');
recordKeys(startTime,goTime,kbNum);

cd(S.subData); % for saving data

% Loop through stimulus trials
blockTrials = find(theData.block == block);
for Trial = 1:length(blockTrials)
    
    % determine specific stim num
    stimNum = blockTrials(Trial);
    
    % skip foil words
    if strcmp(theData.cond(stimNum),'F')
        continue
    end
    
    string_gen = '';
    string_spec = '';
    FlushEvents ('keyDown');
    ListenChar(2);
    
    goTime = goTime + stimTime;
    theData.onset(stimNum) = GetSecs - startTime;
    
    keys = {'NR'};
    RT = 0;
    
    % Blank
    Screen(Window,'FillRect', S.screenColor);
    
    % Display empty box prompting category choice
    destRect = [xcenter-catChoicewidth/2 ycenter-catChoiceheight/2 xcenter+catChoicewidth/2 ycenter+catChoiceheight/2];
    Screen('DrawTexture',Window,catChoicePtr,[],destRect);
    
    % Draw word
    word = theData.wordName{stimNum}(3:end);
    DrawFormattedText(Window,word,'center',ycenter-(catChoiceheight/2+50),S.textColor);
    theData.wordShown{stimNum} = word;
    Screen(Window,'Flip');
    
    
    % Collect responses during stimulus presentation
    numReturns = 0;
    while 1
        typedInput = GetChar;
        switch(abs(typedInput))
            case{RETURN},
                break;
            case {DELETE},
                if ~isempty(string_spec);
                    string_spec= string_spec(1:length(string_spec)-1);
                    Screen('TextSize',Window);
                    DrawFormattedText(Window,string_spec,'center',yCoor_type,S.textColorResp);
                end;
            otherwise, % all other keys
                string_spec= [string_spec typedInput];
                Screen('TextSize',Window);
                
                % new line if too long!
                if length(string_spec)-(numReturns*lengthLine) > lengthLine
                    numReturns = numReturns +1;
                    string_spec = [string_spec '\n'];
                    DrawFormattedText(Window,string_spec,'center',yCoor_type,S.textColorResp);
                else
                    DrawFormattedText(Window,string_spec,'center',yCoor_type,S.textColorResp);
                end;
                
        end;
        
        %draw word & rec again
        DrawFormattedText(Window,word,'center',ycenter-(catChoiceheight/2+50),S.textColor);
        Screen('DrawTexture',Window,catChoicePtr,[],destRect);
        
        Screen('Flip', Window);
        FlushEvents(['keyDown']);
    end
    
    
    
    RT = GetSecs - theData.onset(stimNum);
    keys = string_spec;
    
    if RT < 0.001 % so that RTs of 0 will stand out if they get averaged accidentally
        RT = 999;
    end
    
    theData.stimResp{stimNum} = keys;
    theData.stimRT{stimNum} = RT;
    theData.corrResp{stimNum} = theData.cond{stimNum};
    
    % Present fixation during ITI
    goTime = 0; % added to start over, since stim time isnt fixed...
    goTime = goTime + nullTime;
    Screen(Window,'FillRect', S.screenColor);
    Screen(Window, 'DrawTexture', fix);
    Screen(Window,'Flip');
    
    % Collect responses during ITI
    [keys RT] = recordKeys(startTime,goTime,S.boxNum);
    
    if RT < 0.001 % so that RTs of 0 will stand out if they get averaged accidentally
        RT = 999;
    end
    
    theData.isiResp{stimNum} = keys;
    theData.isiRT{stimNum} = RT;
    
    % Record trial duration
    theData.dur(stimNum) = (GetSecs - startTime) - theData.onset(stimNum); % duration from stim onset
    
    % Save to mat file
    matName = [saveName '.mat'];
    cmd = ['save ' matName];
    eval(cmd);
end

% % Show fixation (lead-out)
% goTime = 0; %reset again
% goTime = goTime + leadOut;
% Screen(Window,'FillRect', S.screenColor);
% Screen(Window, 'DrawTexture', fix);
% Screen(Window,'Flip');
% recordKeys(startTime,goTime,kbNum);
%
% Screen(Window,'FillRect', S.screenColor);
% Screen(Window,'Flip');
%
% Priority(0);

% Show fixation (lead-out)
goTime = 0; %reset again
goTime = goTime + leadOut;
Screen(Window, 'DrawTexture', blank);
Screen(Window, 'DrawTexture', fix);
Screen(Window,'Flip');
recordKeys(startTime,goTime,kbNum);

Screen(Window, 'DrawTexture', blank);
Screen(Window,'Flip');

Priority(0);
end

function theData = am_study(thePath,listName,S,block,saveName,theData)

%% initialize rand.
rand('twister',sum(100*clock));
kbNum=S.kbNum;

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
cd(thePath.stim);

%% Pre-load images 

% Load fixation
fileName = 'fix.jpg';
pic = imread(fileName);
[fixHeight fixWidth crap] = size(pic);
fix = Screen(Window,'MakeTexture', pic);

% Load blank
fileName = 'blank.jpg';
pic = imread(fileName);
blank = Screen(Window,'MakeTexture', pic);

% Load the stim pictures for the current block
for n = 1:listLength
    if (theData.block(n)==block)
        picname = theData.imgFile{n};  % This is the filename of the image
        pic = imread(picname);
        [imgheight(n) imgwidth(n) crap] = size(pic);
        imgPtrs(n) = Screen('MakeTexture',Window,pic);
    end
end

%% Get everything else ready

% preallocate shit:
trialcount = 0;
for preall = 1:listLength
    if (theData.block(preall)==block)
        theData.onset(preall) = 0;
        theData.dur(preall) =  0;
        theData.stimResp{preall} = 'NR';
        theData.stimRT{preall} = 0;
        theData.isiResp{preall} = 'NR';
        theData.isiRT{preall} = 0;
        theData.picShown{preall} = 'blank';
        theData.wordShown{preall} = 'blank';
    end
end

% get ready screen
Screen(Window, 'DrawTexture', blank);
message = ['STUDY BLOCK ' num2str(block) ' OF 5 \n\n\n'... 
           'For each pair, use your index finger to press button number one \n'... 
           'as soon as you form an association between the word and picture'];
DrawFormattedText(Window,message,'center',ycenter-100,textColor);  
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

% Show fixation (lead-in)
goTime = 0;
goTime = goTime + leadIn;
destRect = [xcenter-fixWidth/2 ycenter-fixHeight/2 xcenter+fixWidth/2 ycenter+fixHeight/2];
Screen(Window, 'DrawTexture', blank);
Screen(Window, 'DrawTexture', fix,[],destRect);
Screen(Window,'Flip');
recordKeys(startTime,goTime,kbNum);

cd(S.subData); % for saving data

% Loop through stimulus trials
for Trial = 1:listLength
    if (theData.block(Trial)==block)
        
        goTime = goTime + stimTime;
        theData.onset(Trial) = GetSecs - startTime;
        
        keys = {'NR'};
        RT = 0;
        
        % Blank
        Screen(Window, 'DrawTexture', blank);
        
        % Draw the word
        Screen(Window,'TextSize', 36);
        word = theData.wordName{Trial}(3:end);
        DrawFormattedText(Window,word,'center',ycenter-220,textColor);
        theData.wordShown{Trial} = word;
        
        % Draw the image
        destRect = [xcenter-imgwidth(Trial)/2 ycenter-imgheight(Trial)/2 xcenter+imgwidth(Trial)/2 ycenter+imgheight(Trial)/2];
        Screen('DrawTexture',Window,imgPtrs(Trial),[],destRect);
        
        % Draw the image name
        Screen(Window,'TextSize', 22);
        word = theData.imgName{Trial};
        theData.picShown{Trial} = word;
        a = find(word=='_');
        word(a) = ' ';
        DrawFormattedText(Window,word,'center',ycenter+150,textColor);
        
        % Flip
        Screen(Window,'Flip');
        
        % Collect responses during stimulus presentation
        [keys RT] = recordKeys(startTime,goTime,S.boxNum);
                
        if RT < 0.001 % so that RTs of 0 will stand out if they get averaged accidentally
            RT = 999;
        end
        
        theData.stimResp{Trial} = keys;
        theData.stimRT{Trial} = RT;
        
        % Present fixation during ITI
        goTime = goTime + nullTime;
        destRect = [xcenter-fixWidth/2 ycenter-fixHeight/2 xcenter+fixWidth/2 ycenter+fixHeight/2];
        Screen(Window, 'DrawTexture', blank);
        Screen(Window, 'DrawTexture', fix,[],destRect);
        Screen(Window,'Flip');
        
        % Collect responses during fixation
        [keys RT] = recordKeys(startTime,goTime,S.boxNum);
        
        if RT < 0.001 % so that RTs of 0 will stand out if they get averaged accidentally
            RT = 999;
        end
        
        theData.isiResp{Trial} = keys;
        theData.isiRT{Trial} = RT;
        
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


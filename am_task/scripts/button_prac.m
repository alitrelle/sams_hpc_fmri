function theData = button_prac(thePath,listName,sNum,S,saveName)

g = mod(sNum,4);
if g == 1
    respCodes = {'Face-Specific' 'Face-General' 'Scene-Specific' 'Scene-General' 'dont-know'};
    labelReminder = 'labelRemind1.jpg';
elseif g == 2
    respCodes = {'Face-General' 'Face-Specific' 'Scene-General' 'Scene-Specific' 'dont-know'};
    labelReminder = 'labelRemind2.jpg';
elseif g == 3
    respCodes = {'Scene-Specific' 'Scene-General' 'Face-Specific' 'Face-General' 'dont-know'};
    labelReminder = 'labelRemind3.jpg';
else
    respCodes = {'Scene-General' 'Scene-Specific' 'Face-General' 'Face-Specific' 'dont-know'};   
    labelReminder = 'labelRemind4.jpg';
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

numPractices = 4;

kbNum = S.kbNum;

% Load fixation
fileName = 'fix.jpg';
pic = imread(fileName);
fix = Screen(Window,'MakeTexture', pic);

% Load blank
fileName = 'blank.jpg';
pic = imread(fileName);
blank = Screen(Window,'MakeTexture', pic);

% Load reminder of key labels
pic = imread(labelReminder);
labelRemind = Screen(Window,'MakeTexture', pic);

% get ready screen
Screen(Window, 'DrawTexture', blank);
message = 'Practice pressing the appropriate buttons';
DrawFormattedText(Window,message,'center','center',textColor);
Screen(Window,'Flip');

% get started
status = 1;
while 1
    getKey('g',S.kbNum);
    if S.scanner == 1
        status = 0;
    else
        S.boxNum = S.kbNum;
        status = 0;
    end
    if status == 0
        break
    end
end

%get cursor out of the way
SetMouse(0,myRect(4));

Priority(MaxPriority(Window));
goTime = 0;
startTime = getSecs;

% show fixation 
goTime = goTime + 2;
Screen(Window, 'DrawTexture', blank);
Screen(Window, 'DrawTexture', fix);
Screen(Window,'Flip');
recordKeys(startTime,goTime,kbNum);

% show labels
goTime = goTime + 2;
Screen(Window, 'DrawTexture', blank);
Screen(Window, 'DrawTexture', labelRemind);
Screen(Window,'Flip');
recordKeys(startTime,goTime,kbNum);

cd(thePath.start);

for Trial = 1:4
    choices = Shuffle(respCodes);
    % show choices again
    goTime = goTime + 5;
    Screen(Window, 'DrawTexture', blank);
    Screen(Window, 'DrawTexture', labelRemind);
    Screen(Window,'Flip');
    recordKeys(startTime,goTime,kbNum);
    for loop = 1:5
        % fix
        goTime = goTime + 1;
        Screen(Window, 'DrawTexture', blank);
        Screen(Window, 'DrawTexture', fix);
        Screen(Window,'Flip');
        recordKeys(startTime,goTime,kbNum);
        % category
        goTime = goTime + 2;
        Screen(Window, 'DrawTexture', blank);
        category = choices{loop};
        DrawFormattedText(Window,category,'center','center',textColor);
        Screen(Window,'Flip');
        [keys RT] = recordKeys(startTime,goTime,S.boxNum);
    end
end

goTime = goTime + 1;
Screen(Window, 'DrawTexture', blank);
Screen(Window, 'DrawTexture', fix);
Screen(Window,'Flip');
recordKeys(startTime,goTime,kbNum);

end

function k = getKeyboardNumberScanner

d=PsychHID('Devices');
k = 0;

for n = 1:length(d)
        if (strcmp(d(n).usageName,'Keyboard'))&&(d(n).version==112); % laptop keyboard
        k = n;
        break
    end
end
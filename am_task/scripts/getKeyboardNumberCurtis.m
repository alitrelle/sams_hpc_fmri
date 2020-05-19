
function k = getKeyboardNumberCurtis

d=PsychHID('Devices');
k = 0;

for n = 1:length(d)
    if (d(n).productID==594)&&(strcmp(d(n).usageName,'Keyboard'));  %560 for Recca % 538 for laptop kb, 544 for alan external, 516 for rm 410
        k=n;
        break
    end
end

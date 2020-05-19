
function k = BH1getKeyboardNumber

d=PsychHID('Devices');
k = 0;

for n = 1:length(d)
    if (strcmp(d(n).usageName,'Keyboard')&&(d(n).vendorID==1452)); % prdocut ID laptop: 560
        k=n;
        break
    end
end
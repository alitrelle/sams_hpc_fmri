
function [deviceNumber boxfound] = getBoxNumber

% Checks the connected USB devices and returns the deviceNumber corresponding
% to the button box we use in the scanner. 
% JC 03/02/06 Wrote it
% JC 09/15/08 Added boxfound
% 3T#2: productID = 686
% 7T: productID = 686

deviceNumber = 0; 
d = PsychHID('Devices');
for n = 1:length(d)
    if (d(n).productID == 686) && (strcmp(d(n).usageName,'Keyboard'));
        deviceNumber = n;
        boxfound = 1;
    end
end
if deviceNumber == 0
    fprintf(['Button box NOT FOUND.\n']);
    boxfound = 0;
end
            
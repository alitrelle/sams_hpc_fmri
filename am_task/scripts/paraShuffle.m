
function varargout = paraShuffle(varargin);

% Shuffles all input lists in the SAME random order.
% Keeps pairs together!
% All lists must be same length.
% JC 01/2006
% AT 04/10/08; Added variable input/output paramaters

newOrder = Shuffle(1:length(varargin{1}));
% newOrder = shuffle(1:length(varargin{1}));

for n=1:nargin
  varargout{n}=varargin{n}(newOrder);
end
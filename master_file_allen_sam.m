close all
clear all
addpath('/Users/administrator/Documents/MATLAB/KiloSort')

% data paths
experimentID = '12222016_277371';
root        = fullfile('/Volumes/data_local_1/corbett/',experimentID);
fname       = [experimentID,'.dat'];
fnameTW     = 'temp_wh.dat'; % will be created. residual of whitened data (not fitting in RAM).

% if chanMap is a string, it will load the channel configuration from a file 
% if chanMap is an array, that will indicate the channel order indexing into the raw data channel order
% This can also be used to drop inactive channels. 

% CHANGE THIS:
numChans = 128;
removeChans = [0,18,63,96];

%ops.chanMap = 1:numChans;
ops.chanMap = [16,112,111,14,110,15,109,12,108,13,107,10,106,11,105,8,104,9,103,6,102,7,101,4,100,5,99,2,98,3,97,96,1,95,62,94,63,93,60,92,61,91,58,90,59,89,56,88,57,87,55,86,54,85,53,84,52,83,51,82,50,81,49,80,48,47,78,46,79,45,76,44,77,43,75,42,74,41,73,40,72,39,71,38,70,37,69,36,68,35,67,34,66,33,65,32,64,31,127,30,126,29,125,28,124,27,123,26,122,25,121,24,120,23,119,22,118,21,117,20,116,19,115,18,114,17,113,128];
ops.chanMap(removeChans+1) = [];
ops.connected = ones(1,numel(ops.chanMap));

ops.NchanTOT    = numChans; % total number of channels
ops.Nchan       = numel(ops.chanMap); % number of active channels 
ops.Nfilt       = 512; % 768 number of filters to use 
ops.Nfilt0      = 512; % 768 number of filters to use 

%%

ops.nfullpasses         = 6; % 6, number of complete passes through data
ops.ForceMaxRAMforDat   = Inf; % if you want to force it to use a swap file on disk and less RAM, or no RAM at all (set to 0). 

% you shouldn't need to change these options
ops.whitening 	= 'full'; % type of whitening, only full for now
ops.learnWU     = 1;      % whether to learn templates
ops.maxFR       = 20000;  % maximum number of spikes to extract per batch
ops.lambda      = 1e3;    % not used
ops.Th0         = -7; % not currently used
ops.ntbuff      = 64;  % samples of symmetrical buffer for whitening and spike detection
ops.scaleproc   = 200; % int16 scaling of whitened data
ops.Nrank       = 3;
ops.NT          = 32 * 1024+ ops.ntbuff; % change the first value (32/64/128) to use less GPU memory

% these options you should know how to change
ops.fs          = 30000; % sampling rate
ops.fshigh      = 300; % high-pass filtering the data above this frequency
ops.verbose     = 1;

% the options might need to be experimented with. 
% where there are more than one value for a parameter (Th, lam, momentum) these indicate start and end values during optimization + (possibly) the value during final spike extraction

ops.Th            = [6 12 6];  % where to set the threshold for spikes 
ops.lam           = [10 40 40]; % how much to force spikes in the same cluster to have the same amplitude (
ops.momentum      = [20 200];   % how many detected spikes to average over (start and end values)
ops.nannealpasses = 20;          % annealing stops after this many passes

% feature extraction options. These features are used in the Phy GUI. 
ops.nNeighPC     = 8; % number of channnels to mask the PCs
ops.nNeigh       = 8; % number of neighboring templates to retain projections of


%%
gpuDevice(1); % resets the GPU, attempting to make as much space available as possible.


%% if you need to reload the data, clear variable 'loaded'
clear loaded
load_data_buff; % loads data into RAM + residual data on SSD

%% if you need to rerun the optimization, clear variable 'initialized'
clear initialized
run_reg_mu; % iterate the template matching (non-overlapping extraction)

%%
fullMPMU; % extracts final spike times (overlapping extraction)

%%
rez.simScore = cr;
rezToPhy2(rez,root)
 
%%
plot_final_waveforms;

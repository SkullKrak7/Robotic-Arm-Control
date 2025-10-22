clear all;
rng(1,'twister');

datasetFolder = 'Speech_Numbers_17000';
ads = audioDatastore(datasetFolder, 'IncludeSubfolders', true, 'FileExtensions', '.wav', 'LabelSource', 'foldernames');

commands = categorical(["one","two","three","four","five","six","seven","eight","nine","zero","background"]);
isCommand = ismember(ads.Labels, commands);
isUnknown = ~ismember(ads.Labels, [commands, "_background_noise_"]);

includeFraction = 1;
mask = rand(numel(ads.Labels), 1) < includeFraction;
isUnknown = isUnknown & mask;
ads.Labels(isUnknown) = categorical("unknown");

adsSubset = subset(ads, isCommand | isUnknown);
countEachLabel(adsSubset)

p1 = 0.6;
p2 = 0.4;
[adsTrain, adsValidation] = splitEachLabel(adsSubset, p1);
numUniqueLabels = numel(unique(adsTrain.Labels));

numTrainFiles = numel(adsTrain.Files);
numValidationFiles = numel(adsValidation.Files);

disp(['Number of training files: ', num2str(numTrainFiles)]);
disp(['Number of validation files: ', num2str(numValidationFiles)]);

fs = 16e3;
segmentDuration = 1;
frameDuration = 0.025;
hopDuration = 0.010;
segmentSamples = round(segmentDuration * fs);
frameSamples = round(frameDuration * fs);
hopSamples = round(hopDuration * fs);
overlapSamples = frameSamples - hopSamples;
FFTLength = 512;
numBands = 50;

afe = audioFeatureExtractor('SampleRate', fs, 'FFTLength', FFTLength, 'Window', hann(frameSamples, 'periodic'), 'OverlapLength', overlapSamples, 'barkSpectrum', true);
setExtractorParameters(afe, 'barkSpectrum', 'NumBands', numBands);

numHops = floor((segmentSamples - overlapSamples) / hopSamples) + 1;

XTrain = zeros(numHops, numBands, 1, numel(adsTrain.Files));
XValidation = zeros(numHops, numBands, 1, numel(adsValidation.Files));

subds = partition(adsTrain, 1, 1);
for idx = 1:numel(subds.Files)
    x = read(subds);
    xPadded = [zeros(floor((segmentSamples - numel(x)) / 2), 1); x; zeros(ceil((segmentSamples - numel(x)) / 2), 1)];
    features = extract(afe, xPadded);
    if size(features, 1) < numHops
        paddedFeatures = zeros(numHops, numBands);
        paddedFeatures(1:size(features, 1), :) = features;
        XTrain(:,:,:,idx) = paddedFeatures;
    else
        XTrain(:,:,:,idx) = features(1:numHops, :);
    end
end

subds = partition(adsValidation, 1, 1);
for idx = 1:numel(subds.Files)
    x = read(subds);
    xPadded = [zeros(floor((segmentSamples - numel(x)) / 2), 1); x; zeros(ceil((segmentSamples - numel(x)) / 2), 1)];
    features = extract(afe, xPadded);
    if size(features, 1) < numHops
        paddedFeatures = zeros(numHops, numBands);
        paddedFeatures(1:size(features, 1), :) = features;
        XValidation(:,:,:,idx) = paddedFeatures;
    else
        XValidation(:,:,:,idx) = features(1:numHops, :);
    end
end

unNorm = 2 / (sum(afe.Window)^2);
epsil = 1e-6;

XTrain = XTrain / unNorm;
XTrain = log10(XTrain + epsil);

XValidation = XValidation / unNorm;
XValidation = log10(XValidation + epsil);

YTrain = removecats(adsTrain.Labels);
YValidation = removecats(adsValidation.Labels);

specMin = min(XTrain, [], 'all');
specMax = max(XTrain, [], 'all');
idx = randperm(numel(adsTrain.Files), 3);
figure('Units', 'normalized', 'Position', [0.2 0.2 0.6 0.6]);
for i = 1:3
    [x, fs] = audioread(adsTrain.Files{idx(i)});
    subplot(2, 3, i)
    plot(x)
    axis tight
    title(string(adsTrain.Labels(idx(i))))
    
    subplot(2, 3, i + 3)
    spect = (XTrain(:,:,1,idx(i))');
    pcolor(spect)
    caxis([specMin specMax])
    shading flat
    
    sound(x, fs)
    pause(2)
end

imageDir = 'SpeechImageData';
if ~exist(imageDir, 'dir')
    mkdir(imageDir);
end

for i = 1:size(XTrain, 4)
    label = YTrain(i);
    labelDir = fullfile(imageDir, 'TrainingData', char(label));
    if ~exist(labelDir, 'dir')
        mkdir(labelDir);
    end
    tmp_image = XTrain(:,:,1,i);
    tmp_image = mat2gray(tmp_image);
    fileName = fullfile(labelDir, ['image' num2str(i) '.png']);
    imwrite(tmp_image, fileName);
end

for i = 1:size(XValidation, 4)
    label = YValidation(i);
    labelDir = fullfile(imageDir, 'ValidationData', char(label));
    if ~exist(labelDir, 'dir')
        mkdir(labelDir);
    end
    tmp_image = XValidation(:,:,1,i);
    tmp_image = mat2gray(tmp_image);
    fileName = fullfile(labelDir, ['image' num2str(i) '.png']);
    imwrite(tmp_image, fileName);
end

numProcessedTrainFiles = numel(adsTrain.Files);
numProcessedValidationFiles = numel(adsValidation.Files);

fprintf('Preprocessing is complete.\n');
fprintf('Number of training files processed: %d\n', numProcessedTrainFiles);
fprintf('Number of validation files processed: %d\n', numProcessedValidationFiles);

clc;
clear all;
close all;

fs = 16000;
recordingDuration = 3;
numRecordings = 40;
pauseDuration = 1;
outputBaseDir = 'AudioDataset';
targetDuration = 1;

classLabels = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'};
for i = 1:length(classLabels)
    classDir = fullfile(outputBaseDir, classLabels{i});
    if ~exist(classDir, 'dir')
        mkdir(classDir);
    end
end

startIndex = 1711;
recorder = audiorecorder(fs, 16, 1);

for digitIdx = 1:length(classLabels)
    digitLabel = classLabels{digitIdx};
    fprintf('Now recording for digit: %s\n', digitLabel);
    
    for recNum = 1:numRecordings
        fprintf('Recording %d of %d for digit %s\n', recNum, numRecordings, digitLabel);
        
        for countdown = 3:-1:1
            fprintf('%d...\n', countdown);
            pause(1);
        end
        
        fprintf('Start speaking now!\n');
        recordblocking(recorder, recordingDuration);
        audioData = getaudiodata(recorder);
        fprintf('Recording completed for this instance.\n');
        
        trimmedAudio = trimAudioToSpeech(audioData, fs, targetDuration);
        
        filename = fullfile(outputBaseDir, digitLabel, sprintf('%s_%d.wav', digitLabel, startIndex + recNum - 1));
        audiowrite(filename, trimmedAudio, fs);
        
        fprintf('Please wait...\n');
        pause(pauseDuration);
    end
    
    fprintf('Finished recording for digit: %s\n\n', digitLabel);
end

fprintf('All recordings completed successfully!\n');
clear recorder;

function trimmedAudio = trimAudioToSpeech(audioData, fs, targetDuration)
    energy = audioData.^2;
    threshold = 0.01 * max(energy);
    speechIndices = find(energy > threshold);
    
    if isempty(speechIndices)
        trimmedAudio = audioData(1:min(targetDuration*fs, length(audioData)));
    else
        startIndex = max(speechIndices(1) - round(0.1 * fs), 1);
        endIndex = min(startIndex + targetDuration * fs - 1, length(audioData));
        trimmedAudio = audioData(startIndex:endIndex);
    end
    
    if length(trimmedAudio) < targetDuration * fs
        trimmedAudio = [trimmedAudio; zeros(targetDuration * fs - length(trimmedAudio), 1)];
    end
end

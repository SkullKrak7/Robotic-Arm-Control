clc;
clear all;
rng(1, 'twister');

dsTrain = imageDatastore('TrainingData', "IncludeSubfolders", true, "LabelSource", "foldernames");
dsVal = imageDatastore('ValidationData', "IncludeSubfolders", true, "LabelSource", "foldernames");

image_size = [99 50 1];

timepoolSize = 12;
YValidation = dsVal.Labels;
num_classes = numel(categories(dsTrain.Labels));
num_filters = 128;
filter_sizes = [3 3];
dropout_rate = 0.2;
learning_rate = 0.001;

layers = [
    imageInputLayer(image_size)
    
    convolution2dLayer(filter_sizes, num_filters, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(filter_sizes, num_filters, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, "Stride", 2)

    convolution2dLayer(filter_sizes, num_filters, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(filter_sizes, num_filters, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, "Stride", 2)

    convolution2dLayer(filter_sizes, num_filters, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(filter_sizes, num_filters, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, "Stride", 2)

    convolution2dLayer(filter_sizes, num_filters, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(filter_sizes, num_filters, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer([timepoolSize, 1], "Stride", 1)
    
    dropoutLayer(dropout_rate)
    ];

analyzeNetwork(layers);

layers = [
    layers

    fullyConnectedLayer(num_classes)
    softmaxLayer
    classificationLayer
    ];

validationPatience = 5;
options = trainingOptions('adam', ...
    "MiniBatchSize", 64, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 5, ...
    'MaxEpochs', 20, ...
    'Shuffle', 'every-epoch', ...
    'L2Regularization', 0.0001, ...
    'ValidationData', dsVal, ...
    'ValidationFrequency', 20, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu', ...
    'InitialLearnRate', 0.0001);

net = trainNetwork(dsTrain, layers, options);

save('trainedSpeechModel.mat', 'net');

YPred = classify(net, dsVal);
confMat = confusionmat(YValidation, YPred);
figure;
heatmap(categories(YValidation), categories(YValidation), confMat);
title('Confusion Matrix for Validation Data');

TP = diag(confMat);
FP = sum(confMat, 2) - TP;
FN = sum(confMat, 1)' - TP;
precision = TP ./ (TP + FP);
recall = TP ./ (TP + FN);
F1 = 2 * (precision .* recall) ./ (precision + recall);
accuracy = sum(YPred == YValidation) / numel(YValidation);

fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f\n', mean(precision));
fprintf('Recall: %.2f\n', mean(recall));
fprintf('F1 Score: %.2f\n', mean(F1));

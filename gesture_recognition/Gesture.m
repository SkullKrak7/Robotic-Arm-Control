clc
clear all
close all
rng(1,'twister');

imds = imageDatastore('NumbersInSignLanguage_16500', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

labelCountBeforeSplit = countEachLabel(imds);
disp('Number of images per label before splitting:');
disp(labelCountBeforeSplit);

[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

labelCountTrain = countEachLabel(imdsTrain);
labelCountVal = countEachLabel(imdsVal);

disp('Number of images per label in training set:');
disp(labelCountTrain);

disp('Number of images per label in validation set:');
disp(labelCountVal);

imageAugmenter = imageDataAugmenter('RandXScale',[0.85 1.35],'RandYScale',[0.85 1.35]);

image_size = [98 50 3];

dsTrain = augmentedImageDatastore(image_size, imdsTrain, 'ColorPreprocessing', 'gray2rgb', 'DataAugmentation', imageAugmenter);
dsVal = augmentedImageDatastore(image_size, imdsVal, 'ColorPreprocessing', 'gray2rgb', 'DataAugmentation', imageAugmenter, 'OutputSizeMode', 'resize');

numTrainImages = numel(dsTrain.Files);
numValImages = numel(dsVal.Files);

disp(['Number of training images after augmentation: ', num2str(numTrainImages)]);
disp(['Number of validation images after augmentation: ', num2str(numValImages)]);

YValidation = imdsVal.Labels;
num_classes = numel(categories(imdsTrain.Labels));
num_filters = 128;
dropout_rate = 0.2;
learning_rate = 0.001;

layers = [
    imageInputLayer([image_size])

    convolution2dLayer([5 5],num_filters,'Padding','same') 
    batchNormalizationLayer
    reluLayer

    convolution2dLayer([5 5],num_filters,'Padding','same') 
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer([5 5],num_filters,'Padding','same') 
    batchNormalizationLayer
    reluLayer

    convolution2dLayer([5 5],num_filters,'Padding','same') 
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer([5 5],num_filters,'Padding','same') 
    batchNormalizationLayer
    reluLayer

    convolution2dLayer([5 5],num_filters,'Padding','same') 
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer([5 5],num_filters,'Padding','same') 
    batchNormalizationLayer
    reluLayer

    convolution2dLayer([5 5],num_filters,'Padding','same') 
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,"Stride",2)

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
"MiniBatchSize",64, ...
'LearnRateSchedule', 'piecewise', ...
'LearnRateDropFactor',0.1, ...
'LearnRateDropPeriod',3, ...
'MaxEpochs',10, ...
'Shuffle','every-epoch', ...
'L2Regularization',0.0001, ...
'ValidationData',dsVal, ...
'ValidationFrequency',20, ...
'Verbose',true, ...
'Plots','training-progress', ...
'ExecutionEnvironment','gpu', ...
'InitialLearnRate', 0.0001, ...
'ValidationPatience', validationPatience);

net = trainNetwork(dsTrain, layers, options);

save('trainedGestureModel.mat', 'net');

YPred = classify(net,dsVal);

YPred_onehot = zeros(numel(YPred), num_classes);
for j = 1:numel(YPred)
    YPred_onehot(j, YPred(j)) = 1;
end

confMat = confusionmat(YValidation, YPred);
figure

heatmap(categories(YValidation), categories(YValidation), confMat);
title('Confusion Matrix for Validation Data')

TP = diag(confMat);
FP = sum(confMat, 2) - TP;
FN = sum(confMat, 1)' - TP;
precision = TP ./ (TP + FP);
recall = TP ./ (TP + FN);
F1 = 2 * (precision .* recall) ./ (precision + recall);
accuracy = sum(YPred == YValidation)/numel(YValidation);

fprintf('Validation Accuracy: %.2f%%\n', accuracy*100);
fprintf('Precision: %.2f\n', mean(precision));
fprintf('Recall: %.2f\n', mean(recall));
fprintf('F1 Score: %.2f\n', mean(F1));

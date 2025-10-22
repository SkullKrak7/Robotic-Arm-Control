clc;
close all;
clear all;

c = webcam;
load('trainedGestureModel.mat');

roiX = 1;     
roiY = 1;     
roiWidth = 200;
roiHeight = 200;

bboxes = [roiX roiY roiWidth roiHeight];

arduinoSerial = serialport('COM4', 57600);

previousGesture = "";

while true
    for countdown = 3:-1:1
        img = snapshot(c);
        annotatedImg = insertText(img, [roiX, roiY], num2str(countdown), ...
                                  'FontSize', 80, 'TextColor', 'red', ...
                                  'BoxColor', 'black', 'BoxOpacity', 0.6);
        imshow(annotatedImg);
        pause(1);
    end

    img = snapshot(c);
    annotatedImg = insertObjectAnnotation(img, 'rectangle', bboxes, 'Processing Area');
    roi = img(roiY:roiY+roiHeight-1, roiX:roiX+roiWidth-1, :);
    roiGray = rgb2gray(roi);
    roiRGB = cat(3, roiGray, roiGray, roiGray);
    roiResized = imresize(roiRGB, [98 50]);

    [predictedIdx, scores] = classify(net, roiResized);
    predictedLabel = string(predictedIdx);
    
    confidenceThreshold = 0.3;

    imshow(annotatedImg);
    hold on;
    rectangle('Position', [roiX, roiY, roiWidth, roiHeight], ...
              'EdgeColor', 'r', 'LineWidth', 2);
    
    if max(scores) >= confidenceThreshold && predictedLabel ~= "unknown"
        if predictedLabel ~= previousGesture {
            writeline(arduinoSerial, num2str(double(predictedIdx) - 1));
            disp(['Gesture label sent: ', num2str(double(predictedIdx) - 1)]);
            previousGesture = predictedLabel;
        end
        
        title(['Predicted Gesture: ', char(predictedLabel)]); 
        hold off;
        drawnow;
    else
        title('Predicted Gesture: Unknown'); 
        hold off;
        drawnow;
    end
    
    pause(0.5);
end

clear c;
clear arduinoSerial;

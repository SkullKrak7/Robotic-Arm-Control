clc;
clear all;
close all;
warning off;

c = webcam;

x = 0;
y = 0;
height = 200;
width = 200;
bboxes = [x y height width];

num_classes = 10;
num_images = 100;

class_names = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'};

for class_idx = 1:num_classes
    output_folder = class_names{class_idx};
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    
    disp(['Capturing images for class: ', output_folder]);
    
    temp = 0;
    tic;
    while temp < num_images
        e = c.snapshot;
        
        IFaces = insertObjectAnnotation(e, 'rectangle', bboxes, 'Processing Area');
        
        imshow(IFaces);
        
        es = imcrop(e, bboxes);
        
        es = imresize(es, [98 50]);
        
        es_gray = rgb2gray(es);
        
        filename = fullfile(output_folder, strcat(num2str(temp), '.bmp'));
        
        imwrite(es_gray, filename);
        
        temp = temp + 1;
        
        drawnow;
    end
    elapsedTime = toc;
    disp(['Time taken to capture 100 images for class ', output_folder, ': ', num2str(elapsedTime), ' seconds']);
end

clear c;

disp('Image capturing complete.');

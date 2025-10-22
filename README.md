# Deep Learning–Based Gesture and Speech Control for Robotic Hand

**MSc Robotics Dissertation Project | University of Sheffield | 2024**

[![MATLAB](https://img.shields.io/badge/MATLAB-R2023a-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![Arduino](https://img.shields.io/badge/Arduino-Nano-00979D.svg)](https://www.arduino.cc/)
[![Gesture Accuracy](https://img.shields.io/badge/Gesture_Accuracy-96.98%25-success.svg)]()
[![Speech Accuracy](https://img.shields.io/badge/Speech_Accuracy-95.03%25-success.svg)]()

## Overview

End-to-end deep learning system achieving **96.98% gesture recognition** and **95.03% speech recognition** accuracy for real-time control of an 11-servo Youbionic Handy Lite robotic hand. Custom CNN architectures integrated with Arduino serial communication enable intuitive human-robot interaction through hand gestures and voice commands.

## Key Achievements

- **Gesture Recognition:** 30-layer CNN with 2.8M parameters, 5×5 filters, 128 channels
- **Speech Recognition:** 30-layer CNN with 1M parameters, Bark-spectrum features, temporal pooling
- **Data Collection:** Automated webcam (100 images/digit) and microphone (40 recordings/digit) acquisition
- **Real-Time Control:** MATLAB-Arduino serial communication at 57,600 baud with confidence thresholding (0.3)
- **Dataset Augmentation:** Added 300 gesture and 250 speech samples per class to public datasets

## System Architecture

### Three-Module Pipeline

**1. Data Acquisition & Preprocessing**
- **Gesture:** Webcam captures 200×200 RGB images → resize to 98×50 → grayscale conversion
- **Speech:** Microphone records 3-second audio (16 kHz) → energy-based trimming to 1 second → Bark-spectrum spectrograms (99×50, 50 frequency bands, 512-point FFT)

**2. Deep Learning Models**

**Gesture CNN:**
- Input: 98×50×3 (RGB converted from grayscale)
- Architecture: 4 blocks × (2 conv layers + batch norm + ReLU + max pool) + dropout (0.2)
- Filters: 128 (5×5), padding: same
- Training: Adam optimiser, piecewise learn rate (0.0001, drop 0.1 every 3 epochs), L2 reg (0.0001), mini-batch 64, 10 epochs

**Speech CNN:**
- Input: 99×50×1 (Bark-spectrum)
- Architecture: 4 blocks × (2 conv layers + batch norm + ReLU + max pool) + temporal pooling (12×1) + dropout (0.2)
- Filters: 128 (3×3), padding: same
- Training: Adam optimiser, piecewise learn rate (0.0001, drop 0.1 every 5 epochs), L2 reg (0.0001), mini-batch 64, 20 epochs

**3. Robotic Control**
- Serial communication (COM4, 57,600 baud) to Arduino Nano
- PWM control of 11× SG90 servos via PCA9685 driver (pulse width 150–600 μs)
- Gesture debouncing to prevent duplicate commands

## Technical Stack

| Category | Technologies |
|----------|--------------|
| **Software** | MATLAB R2023a, Deep Learning Toolbox, Image Processing Toolbox, Audio Toolbox, Arduino IDE |
| **Hardware** | Youbionic Handy Lite, Arduino Nano, PCA9685 16-channel PWM driver, webcam, microphone |
| **ML Architecture** | CNN (30 layers), batch normalisation, dropout, ReLU, max pooling, Adam optimiser |
| **Datasets** | ASL Sign Language (16,500 images), English Spoken Digits (17,000 audio files) |

## Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | Parameters |
|-------|----------|-----------|--------|----------|------------|
| **Gesture Recognition** | **96.98%** | 0.97 | 0.97 | 0.97 | 2.8M |
| **Speech Recognition** | **95.03%** | 0.95 | 0.95 | 0.95 | 1.0M |

### Training Configuration

**Gesture Recognition**
```matlab
Image Size: [98, 50, 3]
Filters: 128 (5×5)
Dropout: 0.2
Learning Rate: 0.0001 (piecewise, drop 0.1 every 3 epochs)
Mini-batch: 64
Max Epochs: 10
Train/Val Split: 80/20
L2 Regularisation: 0.0001
```

**Speech Recognition**
```matlab
Spectrogram Size: [99, 50, 1]
Filters: 128 (3×3)
Dropout: 0.2
Temporal Pooling: 12×1
Learning Rate: 0.0001 (piecewise, drop 0.1 every 5 epochs)
Mini-batch: 64
Max Epochs: 20
Train/Val Split: 60/40
```

## Installation & Usage

### Prerequisites

**Required MATLAB Toolboxes:**
- Deep Learning Toolbox
- Image Processing Toolbox
- Audio Toolbox
- MATLAB Support Package for Arduino Hardware
- MATLAB Support Package for USB Webcams

**Hardware:**
- USB webcam or built-in camera
- Microphone (USB or built-in)
- Arduino Nano (optional, for full robotic control demo)
- Youbionic Handy Lite with PCA9685 driver (optional)

### Quick Start

#### 1. Gesture Recognition

**Collect Training Data:**
```matlab
cd gesture_recognition
run('Webcam_DataAcquisition.m')
```

**Train Model:**
```matlab
run('Gesture.m')
```

**Real-Time Testing:**
```matlab
run('Cam_testing.m')
```

#### 2. Speech Recognition

**Collect Audio Data:**
```matlab
cd speech_recognition
run('Mic_DataAcquisition.m')
```

**Preprocess Audio:**
```matlab
run('dataPreProcessing.m')
```

**Train Model:**
```matlab
run('Speech.m')
```

## File Descriptions

| Script | Purpose | Key Features |
|--------|---------|--------------|
| **Gesture.m** | Train gesture CNN | 80/20 split, random scaling (0.85–1.35), confusion matrix, F1 score |
| **Webcam_DataAcquisition.m** | Capture hand gestures | 200×200 ROI, 98×50 resize, BMP output, 100 images/digit |
| **Cam_testing.m** | Real-time gesture classification | Serial to Arduino, confidence threshold (0.3), debouncing |
| **Speech.m** | Train speech CNN | Bark-spectrum input, temporal pooling (12×1), 20 epochs |
| **Mic_DataAcquisition.m** | Record spoken digits | 16 kHz, 3-second recording, energy-based trimming to 1s |
| **dataPreProcessing.m** | Audio feature extraction | Bark-spectrum (50 bands), 512 FFT, 60/40 train/val split |

## Code Highlights

### Real-Time Gesture Classification
```matlab
% From Cam_testing.m
load('trainedGestureModel.mat');
arduinoSerial = serialport('COM4', 57600);
...
```

### Bark-Spectrum Feature Extraction
```matlab
% From dataPreProcessing.m
afe = audioFeatureExtractor('SampleRate',16e3,'FFTLength',512,'barkSpectrum',true);
```

### CNN Architecture
```matlab
% From Gesture.m
layers = [imageInputLayer([98 50 3]) ... dropoutLayer(0.2) ... classificationLayer];
```

## Dataset Information

### Gesture Recognition
- Source: ASL Sign Language Numbers + custom webcam data
- Size: 16,500 images (1,650 per digit)
- Format: Grayscale BMP, 98×50 pixels
- Augmentation: Random scaling (0.85–1.35)
- Train/Val Split: 80/20

### Speech Recognition
- Source: Free Spoken Digit Dataset + custom recordings
- Size: 17,000 audio files (1,700 per digit)
- Format: WAV, 16 kHz mono, 1-second duration
- Preprocessing: Bark-spectrum spectrograms (99×50)
- Train/Val Split: 60/40

## Arduino Integration

Serial communication sends digit labels (0–9) to Arduino Nano. See dissertation Appendix 6.4 for full servo control code.

## Trained Models

Pre-trained models are not included due to size limits. Run training scripts to generate:
- `trainedGestureModel.mat`
- `trainedSpeechModel.mat`

## Troubleshooting

- Webcam: `webcamlist`
- Arduino: `serialportlist("available")`
- GPU: `gpuDeviceCount`

## Limitations & Future Work

- Speech-to-servo control integration pending
- Explore MFCC + RNNs for speech
- Transfer learning to speed training
- Python port for cross-platform support
- Docker container for reproducibility

## Publications & Documentation

**Full Dissertation:** [docs/230118016_Dissertation.pdf](docs/230118016_Dissertation.pdf)

**Supervisor:** Dr Payam Soulatiantork  
**Institution:** University of Sheffield, Department of Automatic Control and Systems Engineering  (now Dept. of EEE)
**Academic Year:** 2023–2024

## Acknowledgements

- University of Sheffield GPU cluster access
- Dr Payam Soulatiantork for supervision
- Public datasets: [ASL Sign Language](https://www.kaggle.com/datasets/muhammadkhalid/sign-language-for-numbers), [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)
- Tools: MATLAB, Arduino IDE, draw.io, ChatGPT

## License

This project is licensed under the MIT License – see [LICENSE](LICENSE).

## Contact

**Sai Karthik Kagolanu**  
MSc Robotics, University of Sheffield  
GitHub: [@SkullKrak7](https://github.com/SkullKrak7)  
LinkedIn: [linkedin.com/in/sai-karthik-kagolanu](https://linkedin.com/in/sai-karthik-kagolanu)

---

*Built with MATLAB • Trained on GPUs • Deployed on Arduino*
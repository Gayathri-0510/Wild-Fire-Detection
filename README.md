# Real-Time Wildfire Detection and Alert System

## Introduction
Forest wildfires are increasing rapidly and cause severe damage to the environment, wildlife, and human life. Early detection is crucial to minimize destruction and improve emergency response time. Traditional monitoring systems rely heavily on manual supervision and are often slow and inefficient. To address this issue, this project proposes an automated wildfire detection system using deep learning and computer vision techniques to accurately detect fire from images and real-time video streams.

## Abstract
Forest wildfires cause serious environmental and economic damage, making early detection essential. This project presents a Real-Time Fire Detection and Alert System using deep learning and computer vision techniques integrated with surveillance cameras. The system is trained on fire and non-fire image datasets using MobileNetV2 for classification. When fire is detected in live video streams, YOLOv8 is used to identify and localize the fire region within the frame.

The system continuously monitors video input, detects fire in real time, and sends automated alerts with location details to emergency response authorities for immediate action. This approach reduces manual monitoring and provides an efficient solution for smart surveillance and disaster management.

## Project Structure
We have established a comprehensive pipeline:
- `data/`: Raw and processed dataset (from Kaggle wildfire dataset)
- `models/`: Saved MobileNetV2 and YOLOv8 models
- `src/`: Core Python modules for preprocessing, building the model, and detection logic
- `app/`: A Flask-based web interface for manual image testing and viewing streams
- `realtime/`: Implementation of continuous CCTV monitoring
- `config/`: System and environment variables
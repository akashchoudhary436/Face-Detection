# Real-Time Face Landmark Detection and Data Capture

## Overview

This project captures facial landmarks from real-time video, extracts features based on these landmarks, and saves both the landmarks and features to an HDF5 file for later use. The program uses OpenCV for face detection, Dlib for facial landmark detection, and HDF5 for storing data.

## Features

- Real-time face detection and landmark extraction
- Saving of facial landmarks and feature vectors to an HDF5 file
- Visualization of facial landmarks and intermediate points in the video feed

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.x
- CMake
- OpenCV
- Dlib
- Numpy
- HDF5 support (via h5py)

### Installing CMake

Dlib requires CMake for its installation. Follow these steps to download and install CMake:

#### On Windows:

1. Go to the [CMake download page](https://cmake.org/download/).
2. Download the Windows Installer (.msi) for the latest version of CMake.
3. Run the installer and follow the installation instructions. Make sure to select the option to add CMake to the system PATH during installation.

#### On macOS:

1. Clone the Repo.
   ```bash
   git clone https://github.com/akashchoudhary436/Face-Detection.git

2. open folder
   ```bash
   cd FACE-DETECTION

3. Install requirements.txt:

   ```bash
   pip install -r requirements.txt

4. Run the Script:

   ```bash
   python face_detection_app.py
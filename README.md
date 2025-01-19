# Augmented_Reality_Python

Augmented Reality Implementation in Python

## Overview

This project provides an implementation of augmented reality using Python and OpenCV. The main functionalities include drawing borders, mapping images, and drawing cubes in a scene captured by a camera.

## Features

- **Draw Borders**: Detects and draws borders around objects in the scene.
- **Image Mapping**: Maps one image onto another using feature matching and homography.
- **Draw Cubes**: Draws a 3D cube in the scene based on feature points.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ali-rasteh/Augmented_Reality_Python.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Augmented_Reality_Python
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare the datasets and place them in the `DataSets` directory.
2. Run the main script:
   ```bash
   python Codes/main.py
   ```

## Project Structure

- **Codes/**: Contains the main source code files.
  - **main.py**: Main script to run the augmented reality application.
  - **Image_Map.py**: Contains the `Image_Mapper` class for image mapping and drawing functionalities.
  - **Camera_Calib.py**: Contains functions for camera calibration.
  - **AR.py**: Contains functions for drawing various figures (cube, coordinates, circles) in the scene.
- **DataSets/**: Directory to store dataset images for camera calibration and testing.

## Contributors

- Ali Rasteh

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

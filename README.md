# Photo Processing Utilities

This repository contains tools and scripts for photo processing and effects.

## Disposable Camera Effect

The `disposable_camera.ipynb` notebook gives your digital images a nostalgic, disposable camera look.

### Features
*   **High ISO Noise/Grain**: Simulates the grainy texture of film.
*   **Soft Focus**: Adds a subtle blur characteristic of cheap plastic lenses.
*   **Color Grading**: Applies warm tones, faded colors, and specific color shifts (green/yellow/blue).
*   **Vintage Curves**: Flattens highlights and lifts shadows for a faded look.
*   **Vignetting**: Adds darker corners.
*   **Aspect Ratio**: Automatically crops to 4x6 or 6x4.

### Usage
Open the notebook and configure the `IMAGE_PATH` and `OUTPUT_FOLDER` in the "Configuration" section to process your images.

## Object Detection

The `object_detection.ipynb` notebook demonstrates object detection on large images using YOLO models.

### Features
*   **YOLOv11 Integration**: Uses Ultralytics YOLO models (nano and x-large) for detection.
*   **Tiling Strategy**: Splits high-resolution images into tiles to improve detection accuracy on small objects.
*   **Non-Maximum Suppression (NMS)**: Merges detections from multiple tiles and removes duplicates.
*   **Inline Visualization**: Displays processed images with bounding boxes directly in the notebook using Matplotlib.

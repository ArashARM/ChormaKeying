# ChormaKeying with OpenCV

This Python script performs chromakeying (green screen effect) on videos, allowing you to replace a background in a video with a custom image. It uses OpenCV for image and video processing and supports real-time parameter adjustments to fine-tune the chromakeying effect.

**Features**
- Background Replacement: Replace a green screen background with any custom image.
- Real-time Parameter Adjustment: Adjust tolerance, blur size, and the hue/saturation/brightness ranges for chromakeying.
- Mouse Region Selection: Select the green screen region in the first frame of the video to set parameters for the rest of the video.
- Live Preview: See the processed video live as you adjust parameters.
- Save Processed Video: Export the processed video with the new background.
- 
**Requirements**
1. Python 3.x
2. OpenCV (opencv-python and opencv-python-headless)
- You can install the required dependencies via pip:
      pip install opencv-python opencv-python-headless numpy
  
**How to Use**
1. Input Files:
   - VideoPath: Path to the video file with a green screen background.
   - BackGroundPath: Path to the image that will replace the green screen.
   - UpdatedVideoPath: Path to save the processed video with the new background.
  
2. Steps:
   - Run the script to open the video and background image.
   - In the first frame, use the mouse to click on a pixel in the green screen region to set the chromakeying parameters.
   - Adjust the tolerance and blur size sliders to fine-tune the background removal.
   - Press ESC to process the entire video and save the new video with the replaced background.



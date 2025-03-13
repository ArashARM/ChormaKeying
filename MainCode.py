# Enter your code here
import cv2
import numpy as np

buffer = 0
blur_size = 0
window_name = "Chromakeying"
tolerance = 0

VideoPath = 'greenscreen-asteroid.mp4'
#VideoPath = 'greenscreen-demo.mp4'
BackGroundPath = 'RM.jpg'
UpdatedVideoPath = 'NewVideo.mp4'


def extract_neighbour(source, x, y, radius):
    height, width = source.shape[:2]
    x_start = max(x - radius, 0)
    y_start = max(y - radius, 0)
    x_end = min(x + radius + 1, height)
    y_end = min(y + radius + 1, width)
    return source[x_start:x_end, y_start:y_end]


def generate_mask(source, min_hue, max_h, min_sat, min_val):
    global buffer, blur_size

    lower_limit = np.array([min_hue, min_sat, min_val], dtype=np.uint8)
    upper_limit = np.array([max_h, 255, 255], dtype=np.uint8)
    image_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_hsv, lower_limit, upper_limit)
    if blur_size > 0:
        mask = cv2.GaussianBlur(mask, (2 * blur_size + 1, 2 * blur_size + 1), 0)

    return mask


def replace_background(source, new_background, mask):
    mask_float = np.float32(mask) / 255.0
    background_resized = cv2.resize(new_background, (source.shape[1], source.shape[0]))
    background_resized = np.float32(background_resized) / 255.0
    source_resized = np.float32(source) / 255.0
    mask_3d = cv2.merge((mask_float, mask_float, mask_float))
    masked_source = cv2.multiply(source_resized, 1 - mask_3d)
    masked_background = cv2.multiply(background_resized, mask_3d)
    result = cv2.add(masked_source, masked_background)
    return result


def buffer_adjustment(value):
    global buffer
    buffer = value
    buffer = buffer*15

    New_hue_min = 0 if int(hue_min) - int(buffer) < 0 else (hue_min - buffer)
    New_hue_max = 180 if int(hue_max)+ + int(buffer) > 180 else (hue_max + buffer)
    New_sat_min = 0 if int(sat_min) - int(buffer) < 0 else (sat_min - buffer)
    New_val_min = 0 if int(val_min) - int(buffer) < 0 else (val_min - buffer)

    mask = generate_mask(source_image, New_hue_min, New_hue_max, New_sat_min, New_val_min)
    result = replace_background(source_image, new_background, mask)
    cv2.imshow(window_name, result)


def blur_adjustment(value):
    global blur_size
    blur_size = value
    if(blur_size>0):
        mask = generate_mask(source_image, hue_min, hue_max, sat_min, val_min)
        result = replace_background(source_image, new_background, mask)
        cv2.imshow(window_name, result)


def region_selector(action, x, y, flags, userdata):
    global hue_min, sat_min, val_min, hue_max, sat_max, val_max, clean_image,buffer,blur_size
    if action == cv2.EVENT_LBUTTONDOWN:
        clean_image = source_image.copy()
        color_region = extract_neighbour(source_image, x, y, 1)
        color_region_hsv = cv2.cvtColor(color_region, cv2.COLOR_BGR2HSV)
        
        # Extract min and max values for hue, saturation, and value
        hue_min, sat_min, val_min = np.min(color_region_hsv[:, :, 0]), np.min(color_region_hsv[:, :, 1]), np.min(color_region_hsv[:, :, 2])
        hue_max, sat_max, val_max = np.max(color_region_hsv[:, :, 0]), np.max(color_region_hsv[:, :, 1]), np.max(color_region_hsv[:, :, 2])
        
        hue_min = max(0,hue_min-10-buffer)
        hue_max = min(180,hue_max+10+buffer)
        sat_min = max(0,(sat_min/2)-buffer)
        val_min = max(0,(val_min/2)-buffer)

        mask = generate_mask(source_image, hue_min, hue_max, sat_min, val_min)
        result = replace_background(source_image, new_background, mask)
        cv2.imshow(window_name, result)


def get_first_frame_background():
    global source_image, VideoPath, new_background, BackGroundPath
    new_background = cv2.imread(BackGroundPath)
    if new_background is None:
        print("Error loading background image")
        return
    cap = cv2.VideoCapture(VideoPath)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    ret, frame = cap.read()
    source_image = frame
    cap.release()


def setup_video_processing():
    global source_image, new_background
    get_first_frame_background()
    
    clean_image = source_image.copy()
    cv2.imshow(window_name, clean_image)
    cv2.createTrackbar("Tolerance", window_name, buffer, 8, buffer_adjustment)
    cv2.createTrackbar("Mask Blur", window_name, blur_size, 50, blur_adjustment)
    cv2.setMouseCallback(window_name, region_selector)
    cv2.putText(clean_image, "Select parameters on the first frame!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(clean_image, "Select the background patch, then adjust parameters!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(clean_image, "Press ESC to see the processed video!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow(window_name, clean_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def process_video():
    global source_image, new_background, UpdatedVideoPath
    
    cap = cv2.VideoCapture(VideoPath)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(UpdatedVideoPath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        mask = generate_mask(frame, hue_min, hue_max, sat_min, val_min)
        result = replace_background(frame, new_background, mask)
        cv2.imshow("Processed Video", result)
        out.write(result)
        
        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    setup_video_processing()
    process_video()

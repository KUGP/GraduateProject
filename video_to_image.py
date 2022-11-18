import cv2
import numpy as np
import glob
import os

#Make image from video
def video_to_image():
    video_root = 'C:/Users/oem/Desktop/video_data'
    video_path = 'C:/Users/oem/Desktop/video_data/drivingvideo2.mp4'
    cap = cv2.VideoCapture(video_path)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print('width {}\theight {}\ttotal frame {}\tFPS {}'.format(str(width),str(height),str(count),str(fps)))

    num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow('frame',frame)
            path = video_root + '/val'+'/'+f'video_{str(num).zfill(4)}'+'.png'
            cv2.imwrite(path,frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        num += 1
    cap.release()
    cv2.destroyAllWindows()

#make video from image
def image_to_video():
    video_root = 'C:/Users/oem/Desktop/video_data'
    image_path = os.path.join(video_root,'video_image')
    mask_path = os.path.join(video_root,'video_mask')
    img_array = []
    for filename in glob.glob(image_path+'/image_*.png'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(video_root+'/image_video.mp4', cv2.VideoWriter_fourcc(*'FMP4'), 30, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

#make video from mask image
def mask_to_video():
    video_root = 'C:/Users/oem/Desktop/video_data'
    image_path = os.path.join(video_root,'video_image')
    mask_path = os.path.join(video_root,'video_mask')
    img_array = []
    for filename in glob.glob(mask_path+'/mask_*.png'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(video_root+'/mask_video.mp4', cv2.VideoWriter_fourcc(*'FMP4'), 30, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == "__main__":
   image_to_video()
   mask_to_video()


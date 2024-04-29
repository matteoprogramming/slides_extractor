import os
import cv2
import numpy as np


def extract_adjacent_different_images(videopath, min_mean, step_frame):
    print("--------------")
    print("ANLYZING VIDEO")
    print("--------------")
    video = cv2.VideoCapture(videopath)
    if not video.isOpened():
        print("Impossibile to open the video")
        return []
    slides = list()
    ret, prev_frame = video.read()
    slides.append(prev_frame)
    frame_counter = 1
    while ret:
        print(f"Frames processed: {frame_counter} - Slides found: {len(slides)}", end='\r')
        frame_counter += 1
        ret, next_frame = video.read()
        for _ in range(step_frame):
                video.grab()        
        try:
            if ret and next_frame is not None and prev_frame.shape == next_frame.shape:
                diff = cv2.absdiff(prev_frame, next_frame)
                if diff.mean() > min_mean:
                    slides.append(next_frame)
        except cv2.error as e:
            print("Error while calculating the absolute difference:", e)
            continue
        prev_frame = next_frame
    print()
    print("Closing video...")
    print("--------------", end="\n\n")
    video.release()
    return slides


def remove_similar_images(slides, min_mean):
    print("--------------")
    print("Removing garbage")
    print("--------------")
    if len(slides) <= 0:
        print("No images given")
        return []
    unique_slides = list()
    for slide in slides:
        isgood = 1
        for u_slid in unique_slides:
            if u_slid.shape == slide.shape:
                try:
                    diff = cv2.absdiff(slide, u_slid)
                    mean = diff.mean()
                except cv2.error as e:
                    print("Error while calculating the absolute difference:", e)
                    continue
                if mean == 0 or mean < min_mean:
                    isgood *= 0
                    break
            else:
                continue
        if isgood:
            unique_slides.append(slide)
            print(f"Good slides found: {len(unique_slides)}", end="\r")
    print()
    print("--------------", end="\n\n")
    return unique_slides


def auto_crop_image(image, threshold=100):
    _, img_gray = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), threshold, 255, cv2.THRESH_BINARY)
    row_means = np.mean(img_gray, axis=1)
    col_means = np.mean(img_gray, axis=0)
    left = np.argmax(col_means > 0)
    right = np.argmax(col_means[::-1] > 0)
    top = np.argmax(row_means > 0)
    bottom = np.argmax(row_means[::-1] > 0)
    cropped_image = image[top:image.shape[0]-bottom, left:image.shape[1]-right]
    return cropped_image


def save_images(folder_path, images):
    import os
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    for i, image in enumerate(images):
        path = os.path.join(folder_path, f"slide_{i}.jpg")
        print("[SAVED]",path)
        cv2.imwrite(path, image)


def save_images(folder_path, images):
    print("--------------")
    print("SAVING IMAGES")
    print("--------------")
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    for i, image in enumerate(images):
        path = os.path.join(folder_path, f"slide_{i}.jpg")
        print("[SAVED]",path)
        cv2.imwrite(path, image)
    print("--------------", end="\n\n")


def main():
    print("--------------------------")
    print("--------------------------")
    print("|| SLIDES      EXTRACTOR ||")
    print("--------------------------")
    print("--------------------------")
    print()
    min_mean = 1
    video_path = input("Enter the path of the video>")
    folder_path = input("Enter the path of the folder>")
    step_frame = int(input("Enter the step frame (how frames can I jump)>"))
    print("Pay attention! If the result is not good, modify the indicators in the script")
    print()
    slides = extract_adjacent_different_images(video_path, min_mean, step_frame)
    unique_images = remove_similar_images(slides, min_mean)
    unique_images = [auto_crop_image(img) for img in unique_images]
    save_images(folder_path, unique_images)
    

if __name__ == "__main__":

    main()
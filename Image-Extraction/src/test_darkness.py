import os
import cv2
import numpy as np

lobbies = os.listdir('/Users/timkleinlein/2022-02-08_S1')
lobbies = [1,2,3,4,5]

dic = {}


# Coordinates for the bottom-left cropping [x1, y1, x2, y2]
# Adjust these values based on your image dimensions and area of interest
bottom_left_coords = [20, -50, 100, -1]  # Example: last 100 pixels in height and first 200 in width

# Darkness threshold
DARKNESS_THRESHOLD = 60  # Adjust as needed (0 to 255 scale)


def analyze_darkness(img_path, crop_coords, threshold):
    # Load the image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")

    # Calculate cropping bounds
    x1, y1, x2, y2 = crop_coords
    if y1 < 0: y1 = img.shape[0] + y1  # Adjust negative y1 to start from the bottom
    if y2 < 0: y2 = img.shape[0] + y2  # Adjust negative y2 to end at the bottom

    # Crop the image
    cropped_img = img[y1:y2, x1:x2]
    # cv2.imwrite('/Users/timkleinlein/2022-02-08_S1/2/2022-02-08_S1_irepptar_1291202860/2/relevant_images/1996_cropped.jpg', cropped_img)

    # Calculate the average pixel intensity
    avg_intensity = np.mean(cropped_img)

    # Determine if it's darker or brighter than the threshold
    is_dark = avg_intensity < threshold

    return avg_intensity, is_dark

for l in lobbies:
    dic[l] = {}
    streamer = os.listdir(f'/Users/timkleinlein/2022-02-08_S1/{l}')
    streamer.remove('.DS_Store')
    for s in streamer:
        dic[l][s] = {}
        disc_rounds = os.listdir(f'/Users/timkleinlein/2022-02-08_S1/{l}/{s}')
        if '.DS_Store' in disc_rounds:
            disc_rounds.remove('.DS_Store')
        for dr in disc_rounds:
            if str(int(dr) - 1) in dic[l][s].keys():
                dic[l][s][dr] = dic[l][s][str(int(dr)-1)]
            else:
                dic[l][s][dr] = 0
            if dic[l][s][dr] == 1:
                continue
            else:
                if 'relevant_images' in os.listdir(f'/Users/timkleinlein/2022-02-08_S1/{l}/{s}/{dr}'):
                    image_paths = os.listdir(f'/Users/timkleinlein/2022-02-08_S1/{l}/{s}/{dr}/relevant_images')
                    if '.DS_Store' in image_paths:
                        image_paths.remove('.DS_Store')
                    darkness_values = []
                    for ip in image_paths:
                        avg_intensity, is_dark = analyze_darkness(f'/Users/timkleinlein/2022-02-08_S1/{l}/{s}/{dr}/relevant_images/{ip}', bottom_left_coords, DARKNESS_THRESHOLD)
                        darkness_values.append(avg_intensity)
                    is_dark = (np.median(darkness_values) < DARKNESS_THRESHOLD)
                    if is_dark:
                        dic[l][s][dr] = 1
                    else:
                        dic[l][s][dr] = 0
                else:
                    dic[l][s][dr] = 0









import cv2
import numpy as np

# File path to the image
image_path = '/Users/timkleinlein/2022-02-08_S1/2/2022-02-08_S1_jvckk_1291644745/2/relevant_images/1534.jpg'
image_path2 = '/Users/timkleinlein/2022-02-08_S1/2/2022-02-08_S1_aribunnie_1291641971/2/relevant_images/1283.jpg'

image_path3 = '/Users/timkleinlein/2022-02-08_S1/2/2022-02-08_S1_irepptar_1291202860/2/relevant_images/1996.jpg'
image_path4 = '/Users/timkleinlein/2022-02-08_S1/2/2022-02-08_S1_ozzaworld_1291365190/2/relevant_images/1490.jpg'
image_path5 = '/Users/timkleinlein/2022-02-08_S1/2/2022-02-08_S1_skadj_1291592126/2/relevant_images/1314.jpg'

# Coordinates for the bottom-left cropping [x1, y1, x2, y2]
# Adjust these values based on your image dimensions and area of interest
bottom_left_coords = [20, -50, 100, -1]  # Example: last 100 pixels in height and first 200 in width

# Darkness threshold
DARKNESS_THRESHOLD = 50  # Adjust as needed (0 to 255 scale)


def analyze_darkness(img_path, crop_coords, threshold):
    # Load the image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")

    # Calculate cropping bounds
    x1, y1, x2, y2 = crop_coords
    if y1 < 0: y1 = img.shape[0] + y1  # Adjust negative y1 to start from the bottom
    if y2 < 0: y2 = img.shape[0] + y2  # Adjust negative y2 to end at the bottom

    # Crop the image
    cropped_img = img[y1:y2, x1:x2]
    cv2.imwrite('/Users/timkleinlein/2022-02-08_S1/2/2022-02-08_S1_irepptar_1291202860/2/relevant_images/1996_cropped.jpg', cropped_img)

    # Calculate the average pixel intensity
    avg_intensity = np.mean(cropped_img)

    # Determine if it's darker or brighter than the threshold
    is_dark = avg_intensity < threshold

    return avg_intensity, is_dark


# Analyze darkness in the bottom-left region
avg_intensity, is_dark = analyze_darkness(image_path, bottom_left_coords, DARKNESS_THRESHOLD)
avg_intensity2, is_dark = analyze_darkness(image_path2, bottom_left_coords, DARKNESS_THRESHOLD)
avg_intensity3, is_dark = analyze_darkness(image_path3, bottom_left_coords, DARKNESS_THRESHOLD)
avg_intensity4, is_dark = analyze_darkness(image_path4, bottom_left_coords, DARKNESS_THRESHOLD)
avg_intensity5, is_dark = analyze_darkness(image_path5, bottom_left_coords, DARKNESS_THRESHOLD)

# Print results
print(f"Average Intensity: {avg_intensity}")
print(f"Is the region darker than threshold ({DARKNESS_THRESHOLD})? {'Yes' if is_dark else 'No'}")



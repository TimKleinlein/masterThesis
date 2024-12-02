import cv2

# File paths
input_image_path = '/Users/timkleinlein/2022-01-26_S1/1/2022-01-26_S1_uneasypeasy_1276935530/0/516.jpg'
output_image_path = '/Users/timkleinlein/2022-01-26_S1/1/2022-01-26_S1_uneasypeasy_1276935530/0/516_cropped.jpg'

# Coordinates for cropping [x1, y1, x2, y2]
crop_coords = [240, 25, 1100, 130]

# Load the image
img = cv2.imread(input_image_path)

# Check if the image was successfully loaded
if img is None:
    print(f"Error: Could not load image from {input_image_path}")
else:
    # Crop the image
    x1, y1, x2, y2 = crop_coords
    cropped_img = img[y1:y2, x1:x2]

    # Save the cropped image
    cv2.imwrite(output_image_path, cropped_img)
    print(f"Cropped image saved to {output_image_path}")







import cv2
import pytesseract

# Path to the cropped image
image_path = '/Users/timkleinlein/2022-01-26_S1/1/2022-01-26_S1_uneasypeasy_1276935530/0/516_cropped.jpg'

# Tesseract configuration
TESSERACT_CONFIG = r'--oem 3 --psm 6'  # Adjust as needed
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to consider extracted text valid

def preprocess_image(img_path):
    # Load image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")

    # Thresholding for binarization
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Morphological opening to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return processed_img

def extract_text_with_confidence(img):
    # Use pytesseract to get detailed OCR data
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=TESSERACT_CONFIG)
    extracted_text = ""

    # Iterate over OCR results and filter by confidence
    for i in range(len(ocr_data["text"])):
        if float(ocr_data["conf"][i]) > (CONFIDENCE_THRESHOLD * 100):  # Confidence is a percentage
            extracted_text += ocr_data["text"][i] + " "

    return extracted_text.strip()

# Preprocess the image
processed_img = preprocess_image(image_path)

# Extract text with confidence filtering
extracted_text = extract_text_with_confidence(processed_img)

# Print the extracted text
print("Extracted Text with Confidence > 0.5:")
print(extracted_text)





def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]



levenshtein_distance('who is the impostor?', extracted_text.lower())

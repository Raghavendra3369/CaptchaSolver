from fastapi import FastAPI, status, UploadFile, File
from fastapi.responses import JSONResponse
import pickle
import numpy as np
import cv2

from src.ResponseModel import SolveCaptchaResponse

app = FastAPI()

model = pickle.load(open('captcha_solver_model.pkl', 'rb'))
lb = pickle.load(open('label_binarizer.pkl', 'rb'))


@app.post('/solve_captcha', status_code=status.HTTP_200_OK, response_model=SolveCaptchaResponse)
async def get_files_containing_keyword(captcha: UploadFile = File(...)):
    try:
        contents = await captcha.read()
        predicted_solution = solve_captcha(contents)
        return {'solution': predicted_solution}

    except Exception as e:
        # In case of any error, return a JSON response with the error message
        return JSONResponse(content={"error": str(e)}, status_code=500)


def solve_captcha(captcha_content):
    np_image = np.fromstring(captcha_content, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add some extra padding around the image
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    # threshold the image
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]

    # find the contours
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letter_image_regions = []

    # Now we can loop through each of the contours and extract the letter
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # checking if any counter is too wide
        # if contour is too wide then there could be two letters joined together or are very close to each other
        if w / h > 1.25:
            # Split it in half into two letter regions
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            letter_image_regions.append((x, y, w, h))

    # Sort the detected letter images based on the x coordinate to make sure
    # Sort the six largest contours based on the product of their width and height (area)
    six_largest_contours = sorted(letter_image_regions, key=lambda x: x[2] * x[3], reverse=True)

    # Get the six largest contours from the sorted list
    six_largest_contours = six_largest_contours[:6]

    # we get them from left-to-right so that we match the right image with the right letter
    six_largest_contours = sorted(six_largest_contours, key=lambda x: x[0])

    # Creating an empty list for storing predicted letters
    predictions = []

    # Save out each letter as a single image
    for letter_bounding_box in six_largest_contours:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

        if letter_image.shape[0] == 0 or letter_image.shape[1] == 0:
            print("Error: Empty letter image")
            continue

        letter_image = cv2.resize(letter_image, (30, 30))

        # Turn the single image into a 4d list of images
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # making prediction
        pred = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(pred)[0]
        predictions.append(letter)

    # Print the captcha's text
    captcha_text = "".join(predictions)
    return int(captcha_text)

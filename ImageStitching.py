import numpy as np  # Import NumPy for numerical operations.
import cv2  # Import OpenCV for image processing.
import glob  # Import glob to fetch file paths matching a pattern.
import imutils  # Import imutils for convenience functions.

# Fetch all image file paths from the 'unstitchedImages' folder with the .jpg extension.
image_paths = glob.glob('unstitchedImages/*.jpg')
images = []  # Initialize an empty list to store images.

# Loop through each image path.
for image in image_paths:
    img = cv2.imread(image)  # Read the image from the path.
    images.append(img)  # Add the image to the list.
    cv2.imshow("Image", img)  # Display the image.
    cv2.waitKey(0)  # Wait for a key press before moving to the next image.

# Create an OpenCV stitcher instance.
imageStitcher = cv2.Stitcher_create()

# Perform stitching on the list of images.
error, stitched_img = imageStitcher.stitch(images)

if not error:  # If stitching is successful (error == 0):
    # Save the stitched image to disk.
    cv2.imwrite("stitchedOutput.png", stitched_img)
    cv2.imshow("Stitched Img", stitched_img)  # Display the stitched image.
    cv2.waitKey(0)  # Wait for a key press.

    # Add a black border of 10 pixels to the stitched image.
    stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

    # Convert the stitched image to grayscale.
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding to create a binary mask of the image.
    thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    cv2.imshow("Threshold Image", thresh_img)  # Display the binary thresholded image.
    cv2.waitKey(0)  # Wait for a key press.

    # Find external contours in the binary image.
    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)  # Simplify the contour data structure.
    # Find the largest contour by area.
    areaOI = max(contours, key=cv2.contourArea)

    # Create a mask of the same size as the threshold image.
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    # Draw a bounding rectangle around the largest contour on the mask.
    x, y, w, h = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # Copy the mask for further processing.
    minRectangle = mask.copy()
    sub = mask.copy()

    # Erode the mask iteratively to find the minimum enclosing rectangle.
    while cv2.countNonZero(sub) > 0:
        minRectangle = cv2.erode(minRectangle, None)  # Perform erosion.
        sub = cv2.subtract(minRectangle, thresh_img)  # Subtract the eroded mask from the original.

    # Find contours again on the eroded rectangle mask.
    contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)  # Simplify contour data.
    # Find the largest contour by area.
    areaOI = max(contours, key=cv2.contourArea)

    cv2.imshow("minRectangle Image", minRectangle)  # Display the minimum enclosing rectangle.
    cv2.waitKey(0)  # Wait for a key press.

    # Get the bounding rectangle of the largest contour.
    x, y, w, h = cv2.boundingRect(areaOI)

    # Crop the stitched image to remove unwanted areas outside the bounding rectangle.
    stitched_img = stitched_img[y:y + h, x:x + w]

    # Save the processed stitched image to disk.
    cv2.imwrite("stitchedOutputProcessed.png", stitched_img)

    cv2.imshow("Stitched Image Processed", stitched_img)  # Display the processed stitched image.
    cv2.waitKey(0)  # Wait for a key press.

else:
    # Print an error message if stitching fails.
    print("Images could not be stitched!")
    print("Likely not enough keypoints being detected!")
import cv2


def image_preprocessing(image_path):
    # load the image
    image = cv2.imread(image_path)

    ## Original Image
    # plt.imshow(image)
    # plt.title('Original Image')
    # plt.show()

    # resize the image
    image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_LINEAR)

    # Convert to grad scale
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    # Apply blur filter and binary threshold
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    threshImg = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)[1]

    # f, ax = plt.subplots(1,2)

    # # show the image after appying
    # ax[0].imshow(threshImg)
    # ax[0].set_title('Threshold')

    # Calculate percentage of noise (background)
    noise = cv2.countNonZero(threshImg)
    total_noise = threshImg.shape[0] * threshImg.shape[1]
    noise_ratio = noise / total_noise

    areas = []
    if noise_ratio >= 0.5:
        # Find contours
        contours = cv2.findContours(threshImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

        # sort the contours found, descending
        cnt = sorted(contours, key=cv2.contourArea, reverse=True)

        # calculate the area of each contour found
        areas = sorted([cv2.contourArea(c) for c in contours], reverse=True)
        if areas[0] < 20000:
            return image

        # region of interest (second max area)
        roi = cnt[1]

        # show the image after appying
        # img = cv2.drawContours(image.copy(), [roi], -1, (0,255,0), 2) # (image, contours, contourIdx, color, thickness)
        # ax[1].imshow(img)
        # ax[1].set_title('Detect ROI')
        # plt.show()

        # Crop the image
        x, y, w, h = cv2.boundingRect(roi)
        crop_img = image[y:y + h, x:x + w]

    else:
        return image

    # Return cropped image
    return crop_img
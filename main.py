import cv2
import numpy as np

if __name__ == '__main__':
    image = cv2.imread('input/i3.jpg')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.erode(bw, kernel)
    bw = cv2.dilate(bw, kernel)

    bw, contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    moments = [cv2.moments(c) for c in contours]
    centroids = [(m['m10'] / m['m00'], m['m01'] / m['m00']) for m in moments]

    out_size = 500
    output = np.zeros((out_size, out_size, 3))
    output_points = np.float32([[0, out_size], [0, 0], [out_size, 0]])

    # Pixel half size, manually estimated from input photo
    pixel_half = 8
    delta = 0.7

    for x, y in centroids:
        source_points = np.float32([[x - pixel_half + delta, y + pixel_half],
                                    [x - pixel_half + delta, y - pixel_half],
                                    [x + pixel_half + delta * 2, y - pixel_half]])
        transform = cv2.getAffineTransform(source_points, output_points)
        output += cv2.warpAffine(image, transform, (out_size, out_size))

    output /= len(centroids)
    output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)

    cv2.imwrite('output/o3.png', output)

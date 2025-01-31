import cv2
import numpy as np
import pandas as pd
import json

def load_image(file_path):
    """ Load an image from file. """
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"File not found: {file_path}")
    return image


def geographic_to_pixel(top_right, bottom_left, pos, lat_ratio, lng_ratio):
    """ Convert geographic coordinates (lat/lon) to pixel coordinates. """
    col = int((pos[1] - bottom_left[1]) / lng_ratio)
    row = int((top_right[0] - pos[0]) / lat_ratio)
    return col, row


def normalize_gsd(image, lat_ratio, lng_ratio, target_gsd=0.5):
    """ Normalize GSD (Ground Sampling Distance). """
    current_gsd = (lat_ratio + lng_ratio) / 2
    scale_factor = max(current_gsd / target_gsd, 0.1)  # Prevent tiny values

    new_width = max(1, int(image.shape[1] * scale_factor))
    new_height = max(1, int(image.shape[0] * scale_factor))

    print(f"Scale Factor: {scale_factor}, New Dimensions: {new_width}x{new_height}")

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC), scale_factor


def get_direction_for_img_coord(start, end):
    """ Compute the rotation angle for alignment. """
    vec = np.array(end) - np.array(start)
    vec[1] = -vec[1]

    if vec[0] > 0:
        angle = np.arctan(vec[1] / vec[0]) / 1.57 * 90
    elif vec[0] < 0:
        angle = np.arctan(vec[1] / vec[0]) / 1.57 * 90 + 180
    else:
        angle = 90 if np.sign(vec[1]) == 1 else 270

    return (360 - angle + 90) % 360


def rotate_view_area_and_coordinates(image, view_area_coordinates, angle):
    """ Rotate the image and view area coordinates. """
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    view_area_coordinates = np.hstack([view_area_coordinates, np.ones((view_area_coordinates.shape[0], 1))])
    rotated_view_area_coordinates = (rotation_matrix @ view_area_coordinates.T).T[:, :2]

    return rotated_image, rotated_view_area_coordinates


def compute_bounding_box(image, view_area_coordinates):
    """ Compute the bounding box for cropping. """
    x_min = max(0, np.min(view_area_coordinates[:, 0]).astype(int))
    x_max = min(image.shape[1], np.max(view_area_coordinates[:, 0]).astype(int))
    y_min = max(0, np.min(view_area_coordinates[:, 1]).astype(int))
    y_max = min(image.shape[0], np.max(view_area_coordinates[:, 1]).astype(int))

    return x_min, x_max, y_min, y_max


def histogram_equalization(image):
    """ Apply histogram equalization for contrast normalization. """
    if len(image.shape) == 3 and image.shape[2] == 3:  # RGB Image
        for i in range(3):
            image[:, :, i] = cv2.equalizeHist((image[:, :, i] * 255).astype(np.uint8)) / 255.0
    else:  # Grayscale
        image = cv2.equalizeHist((image * 255).astype(np.uint8)) / 255.0

    return image


def process_image(result, image_dir, output_size=(224, 224)):
    """ Full pipeline to process an image based on metadata. """

    # Load the image
    file_path = f"{image_dir}/{result['map_name']}.tif"
    image = load_image(file_path)

    # Load view area and GPS coordinates
    top_right = json.loads(result['gps_top_right'])
    bottom_left = json.loads(result['gps_botm_left'])
    lat_to_pixel, lng_to_pixel = result['lat_ratio'], result['lng_ratio']
    view_areas = json.loads(result["view_areas"])
    view_area = view_areas[6]

    #target_gsd = max(1e-5, (lat_to_pixel + lng_to_pixel) * 10)  # Adjust based on dataset
    #image, scale_factor = normalize_gsd(image, lat_to_pixel, lng_to_pixel, target_gsd=target_gsd)

    # Convert geographic coordinates to pixel coordinates
    view_area_coordinates = np.array(
        [geographic_to_pixel(top_right, bottom_left, coord, lat_to_pixel, lng_to_pixel) for coord in view_area],
        dtype=np.float32
    ) #* scale_factor

    # Compute the rotation angle
    angle = get_direction_for_img_coord(view_area_coordinates[-1], view_area_coordinates[0])

    # Rotate the image and coordinates
    rotated_image, rotated_view_area_coordinates = rotate_view_area_and_coordinates(image, view_area_coordinates, angle)

    # Compute bounding box
    x_min, x_max, y_min, y_max = compute_bounding_box(rotated_image, rotated_view_area_coordinates)

    # Crop the rotated image
    cropped_image = rotated_image[y_min:y_max, x_min:x_max]

    # Normalize pixel intensities
    cropped_image = cropped_image.astype(np.float32) / 255.0

    # Apply histogram equalization
    cropped_image = histogram_equalization(cropped_image)

    # Resize to fixed size
    final_image = cv2.resize(cropped_image, output_size, interpolation=cv2.INTER_CUBIC)

    return final_image


if __name__ == '__main__':
    # Load train data
    train_df = pd.read_csv('train_data')

    # Find the last row where 'map_name' is 2128
    result = train_df[train_df['map_name'] == 2128].iloc[-1]

    # Process image
    image_dir = "/Users/arman/Desktop/UTA/Thesis/DATA/train_images"
    processed_image = process_image(result, image_dir)

    # Display final processed image
    cv2.imshow("Normalized View Area", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
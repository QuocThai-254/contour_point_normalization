# =========================================================
# Created by Nguyen Quoc Thai
# Date: 28/04/2024
# Description:
#     This file provide functions for Contour Normalization task.
#     Functions provides:
#       create_parser(): Parser for main function.
#       read_and_scale_down_img(path, mask=0, scale_percent=40): Using OpenCV to read the image and scale down the image's size.
#       sqr(x): Perform square operation.
#       euclid_dist_sqr(p1, p2): Perform the euclid distance square operation.
#       class Line(): Line class for the line that is made by 2 points.
#       reduce_contours_half(contours_retains, start_idx, end_idx, point_retains): Reduce points of the half contour base on the Ramer-Douglas-Peucker algorithm.
#       reduce_contours(contours, num_points): Reduce points of the contour base on the Ramer-Douglas-Peucker algorithm.
#       retain_increase_contours_part(contours_retains, points_interp, start_idx, end_idx, point_retains): Retains number of points after perform the point interpolation.
#       generate_contour_points(contours, num_points, option="cubicspline"): Generate the point with the point interpolation.
#       increase_contours(contours, num_points, inter_option, reduction_option="tb"): Increase the point of the contours as the need of input
#       draw_contour(img, mask, num_points, args): Draw the contour for object with predefined number of point and binary mask.
#       main(): main function perform task.
# =========================================================
import cv2
import math
import argparse
import numpy as np
from scipy.interpolate import CubicSpline, interp1d

def create_parser():
    """Parser for main function

    Returns:
        args: all arguments for main function.
    """
    parser = argparse.ArgumentParser(description='Parser for contour point normalization')
    parser.add_argument('--point_number', type=int, help='Number of needed point', default=20)
    parser.add_argument('--img', type=str, help='Path to image', default='../imgs/car.png')
    parser.add_argument('--img_scale', type=int, help='Scale percentage for the img', default=40)
    parser.add_argument('--binary_mask', type=str, help='Path to binary mask', default='../imgs/mask.png')
    parser.add_argument('--print_coor', type=int, help='Print the coordinates on terminal', default=0)

    subparsers = parser.add_subparsers()
    # Subparser for the positional argument
    parser_in = subparsers.add_parser('increase_point', help='Optional arguments for increase_point')
    parser_in.add_argument('--inter_option', type=str, help='Interpolation method: cubicspline, ln, 1d', default='cubicspline')
    parser_in.add_argument('--reduction_option', type=str, help='Reduction option method: tb, lr, 4p', default='tb')

    args = parser.parse_args()
    return args

def read_and_scale_down_img(path, mask=0, scale_percent=40):
    """Using OpenCV to read the image and scale down the image's size.

    Args:
        path (str): path to the image.
        mask (int, optional): Set to 1 if this is the path to the binary mask image. Defaults to 0.
        scale_percent (int, optional): Scale percentage for the image. Defaults to 0.

    Returns:
        resized (np.array): A NumPy array representing the image read from the file.
    """
    if mask:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.bitwise_not(img)
    else:
        img = cv2.imread(path)
    # (h, w, c) = img.shape
    width = int(img.shape[1] * scale_percent / 100) 
    height = int(img.shape[0] * scale_percent / 100) 
    dim = (width, height) 

    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    return resized

def sqr(x):
    """Perform square operation.

    Args:
        x (any): The number for the operation.

    Returns:
        any: The result of the operation.
    """
    return x*x

def euclid_dist_sqr(p1, p2):
    """Perform the euclid distance square operation.

    Args:
        p1 (tuple): coordinate of the first point.
        p2 (tuple): coordinate of the second point.

    Returns:
        any: The distance square between 2 points.
    """
    return sqr(p1[0] - p2[0]) + sqr(p1[1] - p2[1])

class Line(object):
    """Line class for the line that is made by 2 points.

    """
    def __init__(self, p1, p2):
        """Initialization for the line

        Args:
            p1 (any): x coordinate of the point.
            p2 (any): y coordinate of the point.
        """
        self.p1 = p1
        self.p2 = p2
        self.lengthSquared = euclid_dist_sqr(self.p1, self.p2)

    def find_nearest_head(self, point):
        """Check if the provided point is near to which head.

        Args:
            point (tuple): coordinate of the point.

        Returns:
            (float): a float value represent for the projection of the point compare to the line.
            "value" < 0 : projection is near to the head p1.
            "value" > 1: projection is near to the head p2.
            0 < "value" < 1: projection is on the line.
        """
        segmentLength = self.lengthSquared
        if segmentLength == 0:
            return euclid_dist_sqr(point, self.p1);
        return ((point[0] - self.p1[0]) * (self.p2[0] - self.p1[0]) + \
        (point[1] - self.p1[1]) * (self.p2[1] - self.p1[1])) / segmentLength

    def dist_sqr(self, point):
        """Calculate the distance square from the provided point to the line.

        Args:
            point (tuple): coordinate of the point.

        Returns:
            (float): the distance squared from the point to the line.
        """
        t = self.find_nearest_head(point)
        # if t < 0:
        #     return euclid_dist_sqr(point, self.p1)
        # if t > 1:
        #     return euclid_dist_sqr(point, self.p2)
        return euclid_dist_sqr(point, [
            self.p1[0] + t * (self.p2[0] - self.p1[0]),
            self.p1[1] + t * (self.p2[1] - self.p1[1])
        ])

    def distanceTo(self, point):
        """Calculate the distance from the provided point to the line.

        Args:
            point (tuple): coordinate of the point.

        Returns:
            (float): the distance from the point to the line.
        """
        return math.sqrt(self.dist_sqr(point))

def reduce_contours_half(contours_retains, start_idx, end_idx, point_retains):
    """Reduce points of the half contour base on the Ramer-Douglas-Peucker algorithm.

    Args:
        contours_retains (np.array): the numpy array contains the coordinates of each point in the contour.
        start_idx (int): The start index of the contour need to be processed.
        end_idx (int): The end index of the contour need to be processed.
        point_retains (int): The number of points need to be retained after reduce.

    Returns:
        result (list): The list of coordinates of the retain point after perform the Ramer-Douglas-Peucker algorithm.
    """
    weights = {}
    def filter(start, end):
        """Find the point that has the largest distance to the line made from the start and end index.

        Args:
            start (int): The start index of the contours need to be processed.
            end (int): The end index of the contours need to be processed.
        """
        if (end > start + 1):
            line = Line(contours_retains[start], contours_retains[end])
            maxDist = -1
            maxDistIndex = 0

            for i in range(start + 1, end):
                dist = line.dist_sqr(contours_retains[i])
                if dist > maxDist:
                    maxDist = dist
                    maxDistIndex = i

            # Store the largest distance the the belong coordinate
            weights[maxDist] = contours_retains[maxDistIndex]
            filter(start, maxDistIndex)
            filter(maxDistIndex, end)

    filter(start_idx, end_idx)
    weights_keys = list(weights.keys())
    # Sort by the key to retain the coordinate that have large distance
    weights_keys.sort(reverse=True)

    result = [
        # Only retain point_retains-2 since it will store 2 head of the contour by default
        weights[weights_keys[i]] for i in range(point_retains-2)
    ]
    result.insert(0,contours_retains[start_idx])
    result.append(contours_retains[end_idx])
    return result

def reduce_contours(contours, num_points):
    """Reduce points of the contour base on the Ramer-Douglas-Peucker algorithm.

    Args:
        contours (np.ndarray): The numpy array contains the coordinates of each point in the contour.
        num_points (int): The number of points need to be retained after reduce.

    Returns:
        result (np.ndarray): The ndarray contains coordinates of the retain point after perform the Ramer-Douglas-Peucker algorithm.
    """
    contours_retains = contours.copy()
    # Remove shape 1 of the np ndarray
    contours_retains = np.squeeze(contours_retains)
    length = len(contours_retains)

    # Reduce the contour for each half
    first_half = reduce_contours_half(contours_retains, 0, int(length/2) - 1, int(num_points/2))
    second_half = reduce_contours_half(contours_retains, int(length/2), length-1, num_points - int(num_points/2))
   
    result = []
    result.extend(first_half)
    result.extend(second_half)
    result = np.array(result)
    # Add the dim 1 as the cv2.drawContours need
    result = np.expand_dims(result, axis=1)

    return result

def retain_increase_contours_part(contours_retains, points_interp, start_idx, end_idx, point_retains):
    """Retains number of points after perform the point interpolation.

    Args:
        contours_retains (np.ndarray): the numpy ndarray contains the coordinates of each point in the contour.
        points_interp (np.array): the numpy array contains the coordinates of each point of the point interpolation.
        start_idx (int): The start index of the point interpolation contour need to be processed.
        end_idx (int): The end index of the point interpolation contour need to be processed.
        point_retains (int): The number of points need to be retained after reduce.

    Returns:
        result (list): The list contains the coordinates of points after perform the reduce task with Ramer-Douglas-Peucker algorithm.
    """
    weights = {}
    def filter_decrease(start, end):
        if (end > start + 1):
            line = Line(points_interp[start], points_interp[end])
            maxDist = -1
            maxDistIndex = 0

            for i in range(start + 1, end):
                dist = line.dist_sqr(points_interp[i])
                if dist > maxDist:
                    maxDist = dist
                    maxDistIndex = i
            while maxDist in weights:
                maxDist -= 0.1
            weights[maxDist] = points_interp[maxDistIndex]
            filter_decrease(start, maxDistIndex)
            filter_decrease(maxDistIndex, end)

    filter_decrease(start_idx, end_idx)
    weights_keys = list(weights.keys())
    # Sort by key but ascending since with the interpolation point 
    # need to retain point close to the provided one
    weights_keys.sort()

    result = []
    # Add the provided points to the result first
    result.extend(contours_retains[:])
    point_append = 0

    for i in weights_keys:
        if point_append == point_retains:
            break
        # Check if the point does not exist before in the result list
        if (weights[i] == result).any() == False:
            result.append(np.array(weights[i], dtype='int32'))
            point_append+=1

    return result

def generate_contour_points(contours, num_points, option="cubicspline"):
    """Generate the point with the point interpolation.

    Args:
        contours (np.ndarray): the numpy ndarray contains the coordinates of each point in the contour.
        num_points (int): The number of points need to be retained after reduce.
        option (string): Choose the option for the data generation: cubicspline, ln, 1d. Default is "cubicspline".
    Returns:
        result (list): The list contains the coordinates of points.
    """
    contours_copy = contours.copy()
    contours_copy = np.squeeze(contours_copy)
    # Using dict to sort the point by x coordinate for the interpolation
    point_dict = {}
    for x, y in zip(contours_copy[:,0], contours_copy[:,1]):
        point_dict[x] = y 

    x_values = list(point_dict.keys())
    x_values.sort()
    y_values = [point_dict[i] for i in x_values]
    
    x_values = np.array(x_values)
    y_values = np.array(y_values)

    x_max = np.max(x_values)
    x_min = np.min(x_values)
    # Function of the linear interpolation
    f = interp1d(x_values, y_values)
    # Function of the Cubic Spline interpolation
    cs = CubicSpline(x_values, y_values)

    start = x_min
    step = 1.0
    num = x_max - x_min
    # Generate the x coordinate, number of them is (x_max - x_min)
    x_interp = start + np.arange(0, num) * step
    while len(x_interp) < num_points:
        x_interp = np.vstack((x_interp, x_interp))
    # Base on the option to choose the line interpolation
    if option == "cubicspline": 
        y_interp = cs(x_interp)  
    elif option == "ln":
        y_interp = np.interp(x_interp, x_values, y_values)
    else:
        y_interp = f(x_interp)   

    y_interp = np.array([int(i) for i in y_interp])
    # Stack the x and y coordinate for the reduction
    points_interp = np.vstack((x_interp, y_interp)).T
    result = retain_increase_contours_part(contours_copy, points_interp, 0, len(points_interp)-1, num_points)
    return result

def increase_contours(contours, num_points, inter_option, reduction_option="tb"):
    """Increase the point of the contours as the need of input

    Args:
        contours (np.ndarray): the numpy ndarray contains the coordinates of each point in the contour.
        num_points (int): The number of points need that was predefined.
        reduction_option (str, optional): Choose different approach for the contour reduction after increase. Defaults to "tb".
            "tb": split to half - half: top - bottom
            "lr": split to half - half: left - right
            "4p": split to 4 equal part
    Returns:
        result (np.array): the numpy array contains the coordinates of each point in the contour.
    """
    contours_need = num_points - len(contours) 
    length = len(contours)
    result = []

    if reduction_option == "tb":
        first_half = generate_contour_points(contours[:int(length/2)], 
                                             int(contours_need/2), 
                                             inter_option) 
        second_half = generate_contour_points(contours[int(length/2):], 
                                              contours_need - int(contours_need/2),
                                              inter_option)   
        result.extend(first_half)
        result.extend(second_half)

    elif reduction_option == "lr":
        first_half = contours[int(length*0.25): int(length*0.75)]
        second_half = contours[0: int(length*0.25)]
        second_half = np.vstack((second_half, contours[int(length*0.75): ]))

        first_half = generate_contour_points(contours[:int(length/2)], 
                                             int(contours_need/2), 
                                             inter_option) 
        second_half = generate_contour_points(contours[int(length/2):], 
                                              contours_need - int(contours_need/2),
                                              inter_option)   
        result.extend(first_half)
        result.extend(second_half)

    elif reduction_option == "4p":
        p1 = contours[0: int(length*0.25)]
        p2 = contours[int(length*0.25): int(length*0.5)]
        p3 = contours[int(length*0.5): int(length*0.75)]
        p4 = contours[int(length*0.75): ]
        
        p1_ = generate_contour_points(p1, int(contours_need/4), inter_option) 
        p2_ = generate_contour_points(p2, int(contours_need/4), inter_option) 
        p3_ = generate_contour_points(p3, int(contours_need/4), inter_option) 
        p4_ = generate_contour_points(p4, contours_need - int(contours_need*0.75), inter_option) 
    
        result.extend(p1_)
        result.extend(p2_)
        result.extend(p3_)
        result.extend(p4_)

    result = np.array(result)
    result = np.expand_dims(result, axis=1)
    return result

def draw_contour(img, mask, num_points, args):
    """Draw the contour for object with predefined number of point and binary mask.

    Args:
        img (np.array): A np array representing the image read from the file.
        mask (np.array): A np array representing the binary mask read from the file.
        num_points (int): number of point on the contour.
        args: arguments for process.

    Returns:
        contours_img (np.array): A np array representing the image.
    """
    contours_img = img.copy()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_img[mask==0] = [255,255,255]
    print(f"number of contours: {len(contours)}")
    # In 1 img there can be a lot of object -> process for each of them 
    for i in range(1,len(contours)):
        if len(contours[i] > num_points):
            contours_retains = reduce_contours(contours[i], num_points)

        elif len(contours[i] < num_points):
            contours_retains = increase_contours(contours_retains, 20, 
                                                args.inter_option,
                                                args.reduction_option)

        if args.print_coor == 1:
            print(f"Coordinates of contour: {contours_retains}")

        cv2.drawContours(contours_img, contours_retains, -1, (0,0,255), 5)
    return contours_img

if __name__ == '__main__':
    args = create_parser()
    img = read_and_scale_down_img(args.img, scale_percent=args.img_scale)
    mask = read_and_scale_down_img(args.binary_mask, mask=1, scale_percent=args.img_scale)
    contours_img = draw_contour(img, mask, args.point_number, args)
    cv2.imshow("image", contours_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
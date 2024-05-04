# Contour's point normalization with Ramer-Douglas-Deucker and point interpolation

# Table of content
1. [About](#about)
2. [Set up Environment](#set-up-environment)
3. [How to run](#how-to-run)

## About

This is the code representing for Contour Normalization Algorithm for Segmentation Masks. For more information about the Algorithm setup and explanation, find them in the **pdf folder**.

## Set up Environment

1. Set up the environment with python=3.9 or with conda:
```
conda create -n myenv python=3.9
```

2. Install requirements packages by running the cli:
```
pip install -r requirements.txt
```

Then the environment for the test was already setup.

## How to run

Input of this code can be found in the **imgs folder**, with normal image is *car.png*, binary mask image is *mask.png*

To run the source code of task 1 please run the below cli for more detail:
```
cd task_1_code
python main.py -h
```

Or for the easier way please run as below:
```
python main.py --point_number 20 increase_point --inter_option ln --reduction_option 4p
```

For the positional argument *increase_point*, it must be included for all cli. This cli performs the Contour Normalization task, in case the number of points in the contour is larger than the predefined *point_number* they will use the Ramer-Douglas-Peucker algorithm to reduce. On the opposite, they will split the contour into 4 equal parts and use liner interpolation for each part with the number of additional points is point_number/4.

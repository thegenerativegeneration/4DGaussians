import argparse
import os
import cv2

from tqdm import tqdm

# recreate processing from https://github.com/google/nerfies/blob/main/notebooks/Nerfies_Capture_Processing.ipynb
# intend to add a way to create the points.npy file from the colmap database
import bisect
import json
from pathlib import Path
import cv2

import imageio
from scipy import linalg
from absl import logging
from typing import Dict
import numpy as np
from nerfies.camera import Camera
import pycolmap
from pycolmap import Quaternion
import pandas as pd


def convert_colmap_camera(colmap_camera, colmap_image):
  """Converts a pycolmap `image` to an SFM camera."""
  camera_rotation = colmap_image.R()
  camera_position = -(colmap_image.t @ camera_rotation)
  new_camera = Camera(
      orientation=camera_rotation,
      position=camera_position,
      focal_length=colmap_camera.fx,
      pixel_aspect_ratio=colmap_camera.fx / colmap_camera.fx,
      principal_point=np.array([colmap_camera.cx, colmap_camera.cy]),
      radial_distortion=np.array([colmap_camera.k1, colmap_camera.k2, 0.0]),
      tangential_distortion=np.array([colmap_camera.p1, colmap_camera.p2]),
      skew=0.0,
      image_size=np.array([colmap_camera.width, colmap_camera.height])
  )
  return new_camera


def filter_outlier_points(points, inner_percentile):
  """Filters outlier points."""
  outer = 1.0 - inner_percentile
  lower = outer / 2.0
  upper = 1.0 - lower
  centers_min = np.quantile(points, lower, axis=0)
  centers_max = np.quantile(points, upper, axis=0)
  result = points.copy()

  too_near = np.any(result < centers_min[None, :], axis=1)
  too_far = np.any(result > centers_max[None, :], axis=1)

  return result[~(too_near | too_far)]


def _get_camera_translation(camera):
  """Computes the extrinsic translation of the camera."""
  rot_mat = camera.orientation
  return -camera.position.dot(rot_mat.T)


def _transform_camera(camera, transform_mat):
  """Transforms the camera using the given transformation matrix."""
  # The determinant gives us volumetric scaling factor.
  # Take the cube root to get the linear scaling factor.
  scale = np.cbrt(linalg.det(transform_mat[:, :3]))
  quat_transform = ~Quaternion.FromR(transform_mat[:, :3] / scale)

  translation = _get_camera_translation(camera)
  rot_quat = Quaternion.FromR(camera.orientation)
  rot_quat *= quat_transform
  translation = scale * translation - rot_quat.ToR().dot(transform_mat[:, 3])
  new_transform = np.eye(4)
  new_transform[:3, :3] = rot_quat.ToR()
  new_transform[:3, 3] = translation

  rotation = rot_quat.ToR()
  new_camera = camera.copy()
  new_camera.orientation = rotation
  new_camera.position = -(translation @ rotation)
  return new_camera


def _pycolmap_to_sfm_cameras(manager: pycolmap.SceneManager) -> Dict[int, Camera]:
  """Creates SFM cameras."""
  # Use the original filenames as indices.
  # This mapping necessary since COLMAP uses arbitrary numbers for the
  # image_id.
  image_id_to_colmap_id = {
      image.name.split('.')[0]: image_id
      for image_id, image in manager.images.items()
  }

  sfm_cameras = {}
  for image_id in image_id_to_colmap_id:
    colmap_id = image_id_to_colmap_id[image_id]
    image = manager.images[colmap_id]
    camera = manager.cameras[image.camera_id]
    sfm_cameras[image_id] = convert_colmap_camera(camera, image)

  return sfm_cameras


class SceneManager:
  """A thin wrapper around pycolmap."""

  @classmethod
  def from_pycolmap(cls, colmap_path, image_path, min_track_length=10):
    """Create a scene manager using pycolmap."""
    manager = pycolmap.SceneManager(str(colmap_path))
    manager.load_cameras()
    manager.load_images()
    manager.load_points3D()
    manager.filter_points3D(min_track_len=min_track_length)
    sfm_cameras = _pycolmap_to_sfm_cameras(manager)
    return cls(sfm_cameras, manager.get_filtered_points3D(), image_path)

  def __init__(self, cameras, points, image_path):
    self.image_path = Path(image_path)
    self.camera_dict = cameras
    self.points = points

    logging.info('Created scene manager with %d cameras', len(self.camera_dict))

  def __len__(self):
    return len(self.camera_dict)

  @property
  def image_ids(self):
    return sorted(self.camera_dict.keys())

  @property
  def camera_list(self):
    return [self.camera_dict[i] for i in self.image_ids]

  @property
  def camera_positions(self):
    """Returns an array of camera positions."""
    return np.stack([camera.position for camera in self.camera_list])

  def load_image(self, image_id):
    """Loads the image with the specified image_id."""
    path = self.image_path / f'{image_id}.png'
    with path.open('rb') as f:
      return imageio.imread(f)


  def change_basis(self, axes, center):
    """Change the basis of the scene.

    Args:
      axes: the axes of the new coordinate frame.
      center: the center of the new coordinate frame.

    Returns:
      A new SceneManager with transformed points and cameras.
    """
    transform_mat = np.zeros((3, 4))
    transform_mat[:3, :3] = axes.T
    transform_mat[:, 3] = -(center @ axes)
    return self.transform(transform_mat)

  def transform(self, transform_mat):
    """Transform the scene using a transformation matrix.

    Args:
      transform_mat: a 3x4 transformation matrix representation a
        transformation.

    Returns:
      A new SceneManager with transformed points and cameras.
    """
    if transform_mat.shape != (3, 4):
      raise ValueError('transform_mat should be a 3x4 transformation matrix.')

    points = None
    if self.points is not None:
      points = self.points.copy()
      points = points @ transform_mat[:, :3].T + transform_mat[:, 3]

    new_cameras = {}
    for image_id, camera in self.camera_dict.items():
      new_cameras[image_id] = _transform_camera(camera, transform_mat)

    return SceneManager(new_cameras, points, self.image_path)

  def filter_images(self, image_ids):
    num_filtered = 0
    for image_id in image_ids:
      if self.camera_dict.pop(image_id, None) is not None:
        num_filtered += 1

    return num_filtered
  
def load_colmap_scene(colmap_dir, rgb_dir, colmap_image_scale):
    scene_manager = SceneManager.from_pycolmap(
        colmap_dir / 'sparse/0',
        rgb_dir / f'1x',
        min_track_length=5)

    if colmap_image_scale > 1:
        print(f'Scaling COLMAP cameras back to 1x from {colmap_image_scale}x.')
        for item_id in scene_manager.image_ids:
            camera = scene_manager.camera_dict[item_id]
            scene_manager.camera_dict[item_id] = camera.scale(colmap_image_scale)

def variance_of_laplacian(image: np.ndarray) -> np.ndarray:
    """Compute the variance of the Laplacian which measure the focus."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def filter_blurry(rgb_dir, scene_manager):

    blur_filter_perc = 95.0 # @param {type: 'number'}
    if blur_filter_perc > 0.0:
        image_paths = sorted(rgb_dir.iterdir())
        print('Loading images.')
        images = list(map(scene_manager.load_image, scene_manager.image_ids))
        print('Computing blur scores.')
        blur_scores = np.array([variance_of_laplacian(im) for im in images])
        blur_thres = np.percentile(blur_scores, blur_filter_perc)
        blur_filter_inds = np.where(blur_scores >= blur_thres)[0]
        blur_filter_scores = [blur_scores[i] for i in blur_filter_inds]
        blur_filter_inds = blur_filter_inds[np.argsort(blur_filter_scores)]
        blur_filter_scores = np.sort(blur_filter_scores)
        blur_filter_image_ids = [scene_manager.image_ids[i] for i in blur_filter_inds]
        print(f'Filtering {len(blur_filter_image_ids)} IDs: {blur_filter_image_ids}')
        num_filtered = scene_manager.filter_images(blur_filter_image_ids)
        print(f'Filtered {num_filtered} images')

def estimate_near_far_for_image(scene_manager, image_id):
  """Estimate near/far plane for a single image based via point cloud."""
  points = filter_outlier_points(scene_manager.points, 0.95)
  points = np.concatenate([
      points,
      scene_manager.camera_positions,
  ], axis=0)
  camera = scene_manager.camera_dict[image_id]
  pixels = camera.project(points)
  depths = camera.points_to_local_points(points)[..., 2]

  # in_frustum = camera.ArePixelsInFrustum(pixels)
  in_frustum = (
      (pixels[..., 0] >= 0.0)
      & (pixels[..., 0] <= camera.image_size_x)
      & (pixels[..., 1] >= 0.0)
      & (pixels[..., 1] <= camera.image_size_y))
  depths = depths[in_frustum]

  in_front_of_camera = depths > 0
  depths = depths[in_front_of_camera]

  near = np.quantile(depths, 0.001)
  far = np.quantile(depths, 0.999)

  return near, far


def estimate_near_far(scene_manager):
  """Estimate near/far plane for a set of randomly-chosen images."""
  # image_ids = sorted(scene_manager.images.keys())
  image_ids = scene_manager.image_ids
  rng = np.random.RandomState(0)
  image_ids = rng.choice(
      image_ids, size=len(scene_manager.camera_list), replace=False)

  result = []
  for image_id in image_ids:
    near, far = estimate_near_far_for_image(scene_manager, image_id)
    result.append({'image_id': image_id, 'near': near, 'far': far})
  result = pd.DataFrame.from_records(result)
  return result

def get_bbox_corners(points):
  lower = points.min(axis=0)
  upper = points.max(axis=0)
  return np.stack([lower, upper])

def compute_scene_center_and_scale(scene_manager):
    """Computes the scene center and scale."""

    points = filter_outlier_points(scene_manager.points, 0.95)
    bbox_corners = get_bbox_corners(
        np.concatenate([points, scene_manager.camera_positions], axis=0))

    scene_center = np.mean(bbox_corners, axis=0)
    scene_scale = 1.0 / np.sqrt(np.sum((bbox_corners[1] - bbox_corners[0]) ** 2))

    print(f'Scene Center: {scene_center}')
    print(f'Scene Scale: {scene_scale}')

def save_scene_info( root_dir, scene_scale, scene_center, bbox_corners, near, far):
    
    scene_json_path = root_dir /  'scene.json'
    with scene_json_path.open('w') as f:
        json.dump({
            'scale': scene_scale,
            'center': scene_center.tolist(),
            'bbox': bbox_corners.tolist(),
            'near': near * scene_scale,
            'far': far * scene_scale,
        }, f, indent=2)

    print(f'Saved scene information to {scene_json_path}')

def save_dataset_split(root_dir, scene_manager):
    all_ids = scene_manager.image_ids
    val_ids = all_ids[::20]
    train_ids = sorted(set(all_ids) - set(val_ids))
    dataset_json = {
        'count': len(scene_manager),
        'num_exemplars': len(train_ids),
        'ids': scene_manager.image_ids,
        'train_ids': train_ids,
        'val_ids': val_ids,
    }

    dataset_json_path = root_dir / 'dataset.json'
    with dataset_json_path.open('w') as f:
        json.dump(dataset_json, f, indent=2)

    print(f'Saved dataset information to {dataset_json_path}')

def save_metadata(root_dir, train_ids, val_ids):
    metadata_json = {}
    for i, image_id in enumerate(train_ids):
        metadata_json[image_id] = {
            'warp_id': i,
            'appearance_id': i,
            'camera_id': 0,
        }
    for i, image_id in enumerate(val_ids):
        i = bisect.bisect_left(train_ids, image_id)
        metadata_json[image_id] = {
            'warp_id': i,
            'appearance_id': i,
            'camera_id': 0,
        }

    metadata_json_path = root_dir / 'metadata.json'
    with metadata_json_path.open('w') as f:
        json.dump(metadata_json, f, indent=2)

    print(f'Saved metadata information to {metadata_json_path}')


def save_cameras(root_dir, camera_paths):
   
    test_camera_dir = root_dir / 'camera-paths'
    for test_path_name, test_cameras in camera_paths.items():
        out_dir = test_camera_dir / test_path_name
        out_dir.mkdir(exist_ok=True, parents=True)
        for i, camera in enumerate(test_cameras):
            camera_path = out_dir / f'{i:06d}.json'
            print(f'Saving camera to {camera_path!s}')
            with camera_path.open('w') as f:
                json.dump(camera.to_json(), f, indent=2)
import os
import cv2
from pathlib import Path
import ffmpeg
import numpy as np
import cv2
import imageio
from PIL import Image

def configure_directories(save_dir, capture_name):

    # The root directory for this capture.
    root_dir = Path(save_dir, capture_name)
    # Where to save RGB images.
    rgb_dir = root_dir / 'rgb'
    rgb_raw_dir = root_dir / 'rgb-raw'
    # Where to save the COLMAP outputs.
    colmap_dir = root_dir / 'colmap'
    colmap_db_path = colmap_dir / 'database.db'
    colmap_out_path = colmap_dir / 'sparse'

    colmap_out_path.mkdir(exist_ok=True, parents=True)
    rgb_raw_dir.mkdir(exist_ok=True, parents=True)

    return root_dir, rgb_dir, rgb_raw_dir, colmap_dir, colmap_db_path, colmap_out_path



def flatten_into_images(video_path, rgb_dir, rgb_raw_dir, max_scale):
    fps = -1  # @param {type:'number'}
    target_num_frames = 100 # @param {type: 'number'}

    cap = cv2.VideoCapture(video_path)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_frames < target_num_frames:
        raise RuntimeError(
        'The video is too short and has fewer frames than the target.')

    if fps == -1:
        fps = int(target_num_frames / num_frames * input_fps)
    print(f"Auto-computed FPS = {fps}")

    # @markdown Check this if you want to reprocess the frames.
    overwrite = False  # @param {type:'boolean'}

    if (rgb_dir / '1x').exists() and not overwrite:
        raise RuntimeError(
            f'The RGB frames have already been processed. Check `overwrite` and run again if you really meant to do this.')
    else:
        filters = f"mpdecimate,setpts=N/FRAME_RATE/TB,scale=iw*{max_scale}:ih*{max_scale}"
        tmp_rgb_raw_dir = 'rgb-raw'
        out_pattern = str('rgb-raw/%06d.png')
        os.makedirs(tmp_rgb_raw_dir, exist_ok=True)
        ffmpeg.input(video_path).filter(filters).output(out_pattern, video_bitrate=fps).run()
        os.makedirs(rgb_raw_dir, exist_ok=True)
        os.system(f'rsync -av {tmp_rgb_raw_dir}/ {rgb_raw_dir}/')

def save_image(path, image: np.ndarray) -> None:
  print(f'Saving {path}')
  if not path.parent.exists():
    path.parent.mkdir(exist_ok=True, parents=True)
  with path.open('wb') as f:
    image = Image.fromarray(np.asarray(image))
    image.save(f, format=path.suffix.lstrip('.'))


def image_to_uint8(image: np.ndarray) -> np.ndarray:
  """Convert the image to a uint8 array."""
  if image.dtype == np.uint8:
    return image
  if not issubclass(image.dtype.type, np.floating):
    raise ValueError(
        f'Input image should be a floating type but is of type {image.dtype!r}')
  return (image * 255).clip(0.0, 255).astype(np.uint8)


def make_divisible(image: np.ndarray, divisor: int) -> np.ndarray:
  """Trim the image if not divisible by the divisor."""
  height, width = image.shape[:2]
  if height % divisor == 0 and width % divisor == 0:
    return image

  new_height = height - height % divisor
  new_width = width - width % divisor

  return image[:new_height, :new_width]


def downsample_image(image: np.ndarray, scale: int) -> np.ndarray:
  """Downsamples the image by an integer factor to prevent artifacts."""
  if scale == 1:
    return image

  height, width = image.shape[:2]
  if height % scale > 0 or width % scale > 0:
    raise ValueError(f'Image shape ({height},{width}) must be divisible by the'
                     f' scale ({scale}).')
  out_height, out_width = height // scale, width // scale
  resized = cv2.resize(image, (out_width, out_height), cv2.INTER_AREA)
  return resized

def resize_into_scales(tmp_rgb_raw_dir, rgb_dir):

    image_scales = "1,2,4,8"  # @param {type: "string"}
    image_scales = [int(x) for x in image_scales.split(',')]

    tmp_rgb_dir = Path('rgb')

    for image_path in Path(tmp_rgb_raw_dir).glob('*.png'):
        image = make_divisible(imageio.imread(image_path), max(image_scales))
    for scale in image_scales:
        save_image(
            tmp_rgb_dir / f'{scale}x/{image_path.stem}.png',
            image_to_uint8(downsample_image(image, scale)))

    #!rsync -av "$tmp_rgb_dir/" "$rgb_dir/"
    os.system(f'rsync -av {tmp_rgb_dir}/ {rgb_dir}/')

def extract_features(rgb_dir, colmap_db_path, oclmap_image_scale):
    share_intrinsics = True  # @param {type: 'boolean'}
    assume_upright_cameras = True  # @param {type: 'boolean'}

    # @markdown This sets the scale at which we will run COLMAP. A scale of 1 will be more accurate but will be slow.
    colmap_image_scale = 4  # @param {type: 'number'}
    colmap_rgb_dir = rgb_dir / f'{colmap_image_scale}x'

    # @markdown Check this if you want to re-process SfM.
    overwrite = False  # @param {type: 'boolean'}

    if overwrite and colmap_db_path.exists():
        colmap_db_path.unlink()

    os.system(f'colmap feature_extractor --SiftExtraction.use_gpu 0 --SiftExtraction.upright {int(assume_upright_cameras)} --ImageReader.camera_model OPENCV --ImageReader.single_camera {int(share_intrinsics)} --database_path "{str(colmap_db_path)}" --image_path "{str(colmap_rgb_dir)}"')

def match_features(colmap_db_path, match_method="exhaustive"):

    if match_method == 'exhaustive':
        os.system(f'colmap exhaustive_matcher --SiftMatching.use_gpu 0 --database_path "{str(colmap_db_path)}"')
    else:
    # Use this if you have lots of frames.
        os.system('wget https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin')
        os.system(f'colmap vocab_tree_matcher --VocabTreeMatching.vocab_tree_path vocab_tree_flickr100K_words32K.bin --SiftMatching.use_gpu 0 --database_path "{str(colmap_db_path)}"')

def reconstruct(refine_principal_point = True, min_num_matches = 32, filter_max_reproj_error = 2, tri_complete_max_reproj_error = 2,
                colmap_db_path = 'colmap/database.db', colmap_rgb_dir = 'rgb', colmap_out_path = 'colmap/sparse'):

    os.system(f'colmap mapper --Mapper.ba_refine_principal_point {int(refine_principal_point)} --Mapper.filter_max_reproj_error {filter_max_reproj_error} --Mapper.tri_complete_max_reproj_error {tri_complete_max_reproj_error} --Mapper.min_num_matches {min_num_matches} --database_path "{str(colmap_db_path)}" --image_path "{str(colmap_rgb_dir)}" --output_path "{str(colmap_out_path)}"')
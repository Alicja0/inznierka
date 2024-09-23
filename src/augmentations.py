import numpy as np
from albumentations.core.composition import Compose
from albumentations.augmentations.geometric.transforms import Flip, Affine, ShiftScaleRotate, \
    GridDistortion, OpticalDistortion
from albumentations.augmentations.transforms import GaussNoise
from albumentations.augmentations.blur.transforms import AdvancedBlur


def augment_image(image: np.ndarray, config: dict) -> np.ndarray:
    """
    Apply augmentations to the input image based on the provided configuration.

    Args:
        image (np.ndarray): The input image to be augmented.
        config (dict): The configuration for augmentations.

    Returns:
        np.ndarray: The augmented image.
    """

    augmentations = []

    if config.get('flip', False):
        augmentations.append(Flip())

    if config.get('rotate', False):
        rotation_limit = config.get('rotate_limit', 90)
        augmentations.append(Affine(rotate=(-rotation_limit, rotation_limit)))

    if config.get('shift_scale_rotate', False):
        augmentations.append(ShiftScaleRotate(shift_limit=config.get('shift_limit', 0.0625),
                                              scale_limit=config.get('scale_limit', 0.1),
                                              rotate_limit=config.get('rotate_limit', 45), p=0.5))

    if config.get('grid_distortion', False):
        augmentations.append(GridDistortion(p=0.5))

    if config.get('optical_distortion', False):
        augmentations.append(OpticalDistortion(p=0.5))

    if config.get('blur', False):
        blur_limit = config.get('blur_limit', 7)
        if blur_limit < 3 or blur_limit % 2 == 0 or not isinstance(blur_limit, int):
            raise ValueError("Wrong blur_limit value, expected odd positive integer greater than 1.")
        augmentations.append(AdvancedBlur(blur_limit=(3, blur_limit), p=0.5))

    if config.get('gauss_noise', False):
        augmentations.append(GaussNoise(var_limit=config.get('noise_var_limit', (10.0, 50.0)), p=0.5))

    # Compose the augmentations
    augmentation_pipeline = Compose(augmentations)

    # Apply the augmentations
    augmented = augmentation_pipeline(image=image)

    return augmented['image']

# Example usage:
# config = {
#     'flip': True,
#     'rotate': True,
#     'rotate_limit': 45,
#     'shift_scale_rotate': True,
#     'shift_limit': 0.1,
#     'scale_limit': 0.2,
#     'rotate_limit': 30,
#     'grid_distortion': True,
#     'optical_distortion': True,
#     'blur': True,
#     'blur_limit': 5,
#     'gauss_noise': True,
#     'noise_var_limit': (10.0, 50.0)
# }
# image = np.random.rand(256, 256, 3) * 255  # Example image
# augmented_image = augment_image(image.astype(np.uint8), config)

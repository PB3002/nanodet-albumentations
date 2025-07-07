import cv2
import numpy as np
import albumentations as A
from typing import Dict, Tuple, Optional

from .warp import get_resize_matrix, warp_boxes, get_minimum_dst_shape


class AlbumentationsTransform:
    """A transformation pipeline using Albumentations library while preserving warp_matrix.
    
    Args:
        keep_ratio (bool): Whether to keep aspect ratio when resizing image.
        divisible (int): Make image height and width divisible by this number.
        perspective_prob (float): Probability of applying perspective transform.
        rotate_prob (float): Probability of applying rotation.
        flip_prob (float): Probability of horizontal flip.
        brightness_contrast_prob (float): Probability of adjusting brightness/contrast.
        hue_saturation_prob (float): Probability of adjusting hue/saturation.
        **kwargs: Additional arguments for compatibility.
    """
    def __init__(
        self,
        keep_ratio: bool,
        divisible: int = 0,
        perspective_prob: float = 0.0,
        rotate_prob: float = 0.0,
        flip_prob: float = 0.0,
        brightness_contrast_prob: float = 0.0,
        hue_saturation_prob: float = 0.0,
        **kwargs
    ):
        self.keep_ratio = keep_ratio
        self.divisible = divisible
        
        # Ensure all configuration variables are initialized
        self.affine_cfg = kwargs.get('affine', {})
        self.perspective_cfg = kwargs.get('perspective', {})
        self.coarse_dropout_cfg = kwargs.get('coarse_dropout', {})
        self.blur_cfg = kwargs.get('blur', {})
        self.median_blur_cfg = kwargs.get('median_blur', {})
        self.to_gray_cfg = kwargs.get('to_gray', {})
        self.clahe_cfg = kwargs.get('clahe', {})
        self.iso_noise_cfg = kwargs.get('iso_noise', {})
        self.motion_blur_cfg = kwargs.get('motion_blur', {})
        self.random_gamma_cfg = kwargs.get('random_gamma', {})

        # Initialize probabilities
        self.hue_saturation_prob = kwargs.get('hue_saturation_prob', hue_saturation_prob)
        self.flip_prob = kwargs.get('flip_prob', flip_prob)
        self.perspective_prob = kwargs.get('perspective_prob', perspective_prob)

    def __call__(self, meta_data: Dict, dst_shape: Tuple[int, int]) -> Dict:
        raw_img = meta_data["img"]
        height, width = raw_img.shape[:2]

        # Dynamically set label_fields based on dataset metadata
        label_fields = ['gt_labels'] if 'gt_labels' in meta_data else []

        # Update Albumentations transform pipeline with dynamic label_fields
        self.transform = A.Compose([
            A.HueSaturationValue(
                hue_shift_limit=meta_data.get('hue_shift_limit', 0.015 * 180),
                sat_shift_limit=meta_data.get('sat_shift_limit', 0.7 * 255),
                val_shift_limit=meta_data.get('val_shift_limit', 0.4 * 255),
                p=self.hue_saturation_prob
            ),
            A.Affine(
                rotate=self.affine_cfg.get('rotate', (-3, 3)),
                translate_percent=self.affine_cfg.get('translate_percent', (-0.1, 0.1)),
                scale=self.affine_cfg.get('scale', 0.8),
                shear=self.affine_cfg.get('shear', (-1.0, 1.0)),
                border_mode=cv2.BORDER_CONSTANT if self.affine_cfg.get('border_mode', 'constant') == 'constant' else cv2.BORDER_REFLECT,
                mask_value=self.affine_cfg.get('mask_value', 0),
                p=self.affine_cfg.get('p', 0.6)
            ),
            A.Perspective(
                scale=self.perspective_cfg.get('scale', (0.0, 0.01)),
                p=self.perspective_cfg.get('p', self.perspective_prob),
                border_mode=cv2.BORDER_CONSTANT if self.perspective_cfg.get('border_mode', 'constant') == 'constant' else cv2.BORDER_REFLECT
            ),
            A.HorizontalFlip(p=self.flip_prob),
            A.CoarseDropout(
                num_holes_range=self.coarse_dropout_cfg.get('num_holes_range', (1, 8)),
                hole_height_range=self.coarse_dropout_cfg.get('hole_height_range', (10, 25)),
                hole_width_range=self.coarse_dropout_cfg.get('hole_width_range', (10, 25)),
                fill_value=self.coarse_dropout_cfg.get('fill_value', 0),
                p=self.coarse_dropout_cfg.get('p', 0.3)
            ),
            A.Blur(
                blur_limit=self.blur_cfg.get('blur_limit', (3, 7)),
                p=self.blur_cfg.get('p', 0.01)
            ),
            A.MedianBlur(
                blur_limit=self.median_blur_cfg.get('blur_limit', (3, 7)),
                p=self.median_blur_cfg.get('p', 0.01)
            ),
            A.ToGray(p=self.to_gray_cfg.get('p', 0.01)),
            A.CLAHE(
                clip_limit=self.clahe_cfg.get('clip_limit', 2.0),
                tile_grid_size=tuple(self.clahe_cfg.get('tile_grid_size', (8, 8))),
                p=self.clahe_cfg.get('p', 0.01)
            ),
            A.ISONoise(
                color_shift=self.iso_noise_cfg.get('color_shift', (0.01, 0.05)),
                intensity=self.iso_noise_cfg.get('intensity', (0.1, 0.5)),
                p=self.iso_noise_cfg.get('p', 0.2)
            ),
            A.MotionBlur(
                blur_limit=self.motion_blur_cfg.get('blur_limit', (3, 7)),
                p=self.motion_blur_cfg.get('p', 0.1)
            ),
            A.RandomGamma(
                gamma_limit=self.random_gamma_cfg.get('gamma_limit', (80, 120)),
                p=self.random_gamma_cfg.get('p', 0.2)
            ),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['any_name'],
            min_area=0.0,
            min_visibility=0.15,
            clip=True
        ))

        # Calculate resize matrix first (to preserve warp_matrix functionality)
        if self.keep_ratio:
            dst_shape = get_minimum_dst_shape(
                (width, height), dst_shape, self.divisible
            )
        
        ResizeM = get_resize_matrix((width, height), dst_shape, self.keep_ratio)
        
        # Prepare bboxes for Albumentations format if they exist
        transformed = {}
        if "gt_bboxes" in meta_data and len(meta_data["gt_bboxes"]):
            bboxes = meta_data["gt_bboxes"]
            bbox_classes = meta_data["gt_labels"].tolist()

            # Apply Albumentations transforms
            transformed = self.transform(
                image=raw_img,
                bboxes=np.array(bboxes),
                any_name= bbox_classes
            )
            
            # Update image and bboxes
            raw_img = transformed['image']
            if len(transformed['bboxes']):
                meta_data["gt_bboxes"] = np.array(transformed['bboxes'], dtype=np.float32)
                # Update labels to match the transformed bboxes
                meta_data["gt_labels"] = np.array(transformed['any_name'], dtype=np.int64)
            else:
                # If no bboxes remain after transformation, set empty arrays
                meta_data["gt_bboxes"] = np.zeros((0, 4), dtype=np.float32)
                meta_data["gt_labels"] = np.array([], dtype=np.int64)

        # Apply resize transform and update warp matrix
        img = cv2.resize(raw_img, tuple(dst_shape))
        meta_data["img"] = img
        meta_data["warp_matrix"] = ResizeM

        # Transform bboxes using resize matrix
        if "gt_bboxes" in meta_data:
            meta_data["gt_bboxes"] = warp_boxes(
                meta_data["gt_bboxes"], ResizeM, dst_shape[0], dst_shape[1]
            )
        
        # Handle ignored bboxes if they exist
        if "gt_bboxes_ignore" in meta_data:
            bboxes_ignore = meta_data["gt_bboxes_ignore"]
            meta_data["gt_bboxes_ignore"] = warp_boxes(
                bboxes_ignore, ResizeM, dst_shape[0], dst_shape[1]
            )

        # Handle masks if they exist
        if "gt_masks" in meta_data:
            for i, mask in enumerate(meta_data["gt_masks"]):
                # Apply same Albumentations transform to masks
                transformed_mask = self.transform(image=mask)['image']
                # Then apply resize
                meta_data["gt_masks"][i] = cv2.resize(
                    transformed_mask,
                    tuple(dst_shape),
                    interpolation=cv2.INTER_NEAREST
                )

        return meta_data

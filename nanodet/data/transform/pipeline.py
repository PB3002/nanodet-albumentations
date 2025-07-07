# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import warnings
from typing import Dict, Tuple

from torch.utils.data import Dataset

from .color import color_aug_and_norm
from .warp import ShapeTransform, warp_and_resize
from nanodet.data.transform.albumentations_transform import AlbumentationsTransform


class LegacyPipeline:
    def __init__(self, cfg: Dict, keep_ratio: bool):
        self.shape_transform = ShapeTransform(keep_ratio, **cfg)
        self.color = functools.partial(color_aug_and_norm, kwargs=cfg)

    def __call__(self, dataset: Dataset, meta: Dict, dst_shape: Tuple[int, int]):
        meta = self.shape_transform(meta, dst_shape=dst_shape)
        meta = self.color(meta=meta)
        return meta


class Pipeline:
    """Data process pipeline. Apply augmentation and pre-processing on
    meta_data from dataset.

    Args:
        cfg (Dict): Data pipeline config.
        keep_ratio (bool): Whether to keep aspect ratio when resizing image.
        is_train (bool): Whether in training mode. Training mode uses augmentations.
    """

    def __init__(self, cfg: Dict, keep_ratio: bool, is_train: bool = True):
        self.is_train = is_train
        cfg = {k: v for k, v in cfg.items() if k != 'keep_ratio'}
        
        if self.is_train:
            # Use AlbumentationsTransform for training with all augmentations
            self.shape_transform = AlbumentationsTransform(
                keep_ratio=keep_ratio,
                divisible=cfg.get('divisible', 0),
                perspective_prob=cfg.get('perspective_prob', 0.0),
                rotate_prob=cfg.get('rotate_prob', 0.0),
                flip_prob=cfg.get('flip_prob', 0.0),
                brightness_contrast_prob=cfg.get('brightness_contrast_prob', 0.0),
                hue_saturation_prob=cfg.get('hue_saturation_prob', 0.0),
                affine=cfg.get('affine', {}),
                perspective=cfg.get('perspective', {}),
                coarse_dropout=cfg.get('coarse_dropout', {}),
                blur=cfg.get('blur', {}),
                median_blur=cfg.get('median_blur', {}),
                to_gray=cfg.get('to_gray', {}),
                clahe=cfg.get('clahe', {}),
                iso_noise=cfg.get('iso_noise', {}),
                motion_blur=cfg.get('motion_blur', {}),
                random_gamma=cfg.get('random_gamma', {})
            )
        else:
            # For validation, use warp_and_resize with minimal configuration
            self.warp_kwargs = {
                'warp_kwargs': {
                    'keep_ratio': keep_ratio,
                    'divisible': cfg.get('divisible', 0),
                }
            }
            self.keep_ratio = keep_ratio

        self.color = functools.partial(color_aug_and_norm, kwargs=cfg)

    def __call__(self, dataset: Dataset, meta: Dict, dst_shape: Tuple[int, int]):
        if self.is_train:
            meta = self.shape_transform(meta, dst_shape=dst_shape)
        else:
            # For validation, use warp_and_resize with proper arguments
            meta = warp_and_resize(meta=meta, dst_shape=dst_shape, keep_ratio=self.keep_ratio, warp_kwargs=self.warp_kwargs['warp_kwargs'])
        meta = self.color(meta=meta)
        return meta

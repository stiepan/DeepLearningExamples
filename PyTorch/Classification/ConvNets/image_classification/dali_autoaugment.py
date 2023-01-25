# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import fn
from nvidia.dali import types
from nvidia.dali import tensors

from nvidia.dali.pipeline.experimental import pipeline_def
from nvidia.dali.types import DALIDataType

from functools import wraps
import numpy as np
import torch

# TODO(klecki): TO BE PLACED IN DEDICATED DALI MODULE:
# TODO(klecki): THIS CODE REPLICATES THE DeepLearningExamples parameter range and number of steps
# the AutoAugment paper uses different range!!!!


def op_magnitude_range(lo, hi, num_magnitudes=11, auto_symmetric=False):
    """Decorator for automatizing the the discrete magnitude -> parameter from range mapping.

    Parameters
    ----------
    lo : Float
        Low end of the range
    hi : Float
        High end of the range (inclusive)
    num_magnitudes : int, optional
        How many magnitude steps are there, by default 11
    auto_symmetric : bool, optional
        If this is true, the operation is assumed to accept a range in a form [0, x],
        and the code is automatically transformed, for a given magnitude `m` into
        selecting the m-th value from [0, x], that is:
        `v = np.linspace(0, x, num_magnitudes)[m]`
        and randomly passing `v` or `-v` to the operator.
        By default False

    Returns
    -------
    Callable[[DataNode, MagnitudeIdx, DataNode], DataNode]
        Function that takes as parameters DataNode representing input, the fixed magnitude idx,
        and the potentially unused random variable for symmetric operations (True for positive,
        False for negative).
        Returns a processed DataNode.
    """
    magnitudes = np.linspace(lo, hi, num_magnitudes, dtype=np.float32)

    def decorator(function):

        @wraps(function)
        def wrapper_regular(samples, magnitude_idx, _):
            magnitude = magnitudes[magnitude_idx]
            return function(samples, magnitude)

        @wraps(function)
        def wrapper_symmetric(samples, magnitude_idx, is_positive):
            magnitude = magnitudes[magnitude_idx]
            if is_positive:
                return function(samples, magnitude)
            else:
                return function(samples, -magnitude)

        if auto_symmetric:
            return wrapper_symmetric
        else:
            return wrapper_regular

    return decorator


@op_magnitude_range(0, 0.3, auto_symmetric=True)
def shearX(samples, parameter):
    mt = fn.transforms.shear(shear=[parameter, 0])
    return fn.warp_affine(samples,
                          matrix=mt,
                          fill_value=128,
                          inverse_map=False)


@op_magnitude_range(0, 0.3, auto_symmetric=True)
def shearY(samples, parameter):
    mt = fn.transforms.shear(shear=[0, parameter])
    return fn.warp_affine(samples,
                          matrix=mt,
                          fill_value=128,
                          inverse_map=False)


@op_magnitude_range(0, 250, auto_symmetric=True)
def translateX(samples, parameter):
    mt = fn.transforms.translation(offset=[parameter, 0])
    return fn.warp_affine(samples,
                          matrix=mt,
                          fill_value=128,
                          inverse_map=False)


@op_magnitude_range(0, 250, auto_symmetric=True)
def translateY(samples, parameter):
    mt = fn.transforms.translation(offset=[0, parameter])
    return fn.warp_affine(samples,
                          matrix=mt,
                          fill_value=128,
                          inverse_map=False)


@op_magnitude_range(0, 30, auto_symmetric=True)
def rotate(samples, parameter):
    return fn.rotate(samples, angle=parameter, fill_value=128)


@op_magnitude_range(0.1, 1.9)
def brightness(samples, parameter):
    return fn.brightness(samples, brightness=parameter)


@op_magnitude_range(0.1, 1.9)
def contrast(samples, parameter):
    return fn.contrast(samples, contrast=parameter)


@op_magnitude_range(0.1, 1.9)
def color(samples, parameter):
    return fn.saturation(samples, saturation=parameter)


@op_magnitude_range(0, 4)
def posterize(samples, parameter):
    nbits = np.round(parameter).astype(np.int32)
    mask = np.array(255 ^ (2**(8 - nbits) - 1), dtype=np.uint8)
    return samples & mask


@op_magnitude_range(0, 110)
def solarize(samples, threshold):
    samples_inv = 255 - samples
    mask_left = samples < np.uint8(threshold)
    mask_right = types.Constant(1) - mask_left
    return fn.cast(mask_left * samples + mask_right * samples_inv,
                   dtype=types.UINT8)
    # return samples


@op_magnitude_range(0, 256)
def solarize_add(samples, shift):
    samples_shifted = fn.cast_like(samples + shift, samples)
    mask_left = samples < 128
    mask_right = types.Constant(1) - mask_left
    return fn.cast_like(mask_left * samples_shifted + mask_right * samples,
                        samples)
    # return samples


@op_magnitude_range(0.1, 1.9)
def sharpness(samples, magnitude):
    blur = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=np.float32) / 13
    ident = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    kernel = (1 - magnitude) * blur + magnitude * ident
    return fn.experimental.filter(samples, kernel)


@op_magnitude_range(0, 1)
def invert(samples, _):
    return fn.cast_like(255 - samples, samples)


# TODO(klecki): No-op
@op_magnitude_range(0, 1)
def equalize(samples, _):
    # return fn.experimental.equalize(samples)
    return samples


@op_magnitude_range(0, 1)
def autocontrast(samples, _):
    lo, hi = fn.reductions.min(samples,
                               axes=[-3, -2]), fn.reductions.max(samples,
                                                                 axes=[-3, -2])
    lo = fn.expand_dims(lo, axes=[0, 1])
    hi = fn.expand_dims(hi, axes=[0, 1])
    return fn.cast_like((samples - lo) * (255 / (hi - lo)), samples)


def subpolicy(op_desc0, op_desc1):
    op0, p0, mag0 = op_desc0
    op1, p1, mag1 = op_desc1

    def subpolicy_impl(images, rand0, rand1, is_pos0, is_pos1):
        if rand0 < p0:
            images = op0(images, mag0, is_pos0)
        if rand1 < p1:
            images = op1(images, mag1, is_pos1)
        return images

    return subpolicy_impl


policies = [
    subpolicy((equalize,  0.8, 1), (shearY,       0.8, 4)),
    subpolicy((color,     0.4, 9), (equalize,     0.6, 3)),
    subpolicy((color,     0.4, 1), (rotate,       0.6, 8)),
    subpolicy((solarize,  0.8, 3), (equalize,     0.4, 7)),
    subpolicy((solarize,  0.4, 2), (solarize,     0.6, 2)),
    subpolicy((color,     0.2, 0), (equalize,     0.8, 8)),
    subpolicy((equalize,  0.4, 8), (solarize_add, 0.8, 3)),
    subpolicy((shearX,    0.2, 9), (rotate,       0.6, 8)),
    subpolicy((color,     0.6, 1), (equalize,     1.0, 2)),
    subpolicy((invert,    0.4, 9), (rotate,       0.6, 0)),
    subpolicy((equalize,  1.0, 9), (shearY,       0.6, 3)),
    subpolicy((color,     0.4, 7), (equalize,     0.6, 0)),
    subpolicy((posterize, 0.4, 6), (autocontrast, 0.4, 7)),
    subpolicy((solarize,  0.6, 8), (color,        0.6, 9)),
    subpolicy((solarize,  0.2, 4), (rotate,       0.8, 9)),
    subpolicy((rotate,    1.0, 7), (translateY,   0.8, 9)),
    subpolicy((shearX,    0.0, 0), (solarize,     0.8, 4)),
    subpolicy((shearY,    0.8, 0), (color,        0.6, 4)),
    subpolicy((color,     1.0, 0), (rotate,       0.6, 2)),
    subpolicy((equalize,  0.8, 4), (equalize,     0.0, 8)),
    subpolicy((equalize,  1.0, 4), (autocontrast, 0.6, 2)),
    subpolicy((shearY,    0.4, 7), (solarize_add, 0.6, 7)),
    subpolicy((posterize, 0.8, 2), (solarize,     0.6, 10)),
    subpolicy((solarize,  0.6, 8), (equalize,     0.6, 1)),
    subpolicy((color,     0.8, 6), (rotate,       0.4, 5)),
]

def apply_policies(imgs, policies):
    """To each sample in `imgs` apply the corresponding `policies[policy_id]` """

    policy_id = fn.random.uniform(values=list(range(len(policies))),
                                  dtype=DALIDataType.INT32)

    rand0 = fn.random.uniform()
    rand1 = fn.random.uniform()

    is_pos0 = fn.random.coin_flip(dtype=DALIDataType.BOOL)
    is_pos1 = fn.random.coin_flip(dtype=DALIDataType.BOOL)

    def recursive_split(imgs, rand0, rand1, is_pos0, is_pos1, policy_id,
                        start_offset, elems):
        if elems == 1:
            policy = policies[start_offset]
            return policy(imgs, rand0, rand1, is_pos0, is_pos1)
        half_size = elems // 2
        if policy_id < start_offset + half_size:
            return recursive_split(imgs, rand0, rand1, is_pos0, is_pos1, policy_id,
                                   start_offset, half_size)
        else:
            return recursive_split(imgs, rand0, rand1, is_pos0, is_pos1, policy_id,
                                   start_offset + half_size, elems - half_size)

    return recursive_split(imgs, rand0, rand1, is_pos0, is_pos1, policy_id, 0,
                           len(policies))


@pipeline_def(enable_conditionals=True)
def auto_augment_pipe(data_dir, interpolation, crop, dali_cpu=False, rank=0, world_size=1):
    print("Building DALI with AutoAugment")
    interpolation = {
        "bicubic": types.INTERP_CUBIC,
        "bilinear": types.INTERP_LINEAR,
        "triangular": types.INTERP_TRIANGULAR,
    }[interpolation]

    rng = fn.random.coin_flip(probability=0.5)

    jpegs, labels = fn.readers.file(
        name="Reader",
        file_root=data_dir,
        shard_id=rank,
        num_shards=world_size,
        random_shuffle=True,
        pad_last_batch=True)

    if dali_cpu:
        images = fn.decoders.image(jpegs, device="cpu", output_type=types.RGB)
    else:
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images
        # from full-sized ImageNet without additional reallocations
        images = fn.decoders.image(jpegs,
                                   device="mixed",
                                   output_type=types.RGB,
                                   device_memory_padding=211025920,
                                   host_memory_padding=140544512)

    images = fn.random_resized_crop(
        images,
        size=[crop, crop],
        interp_type=interpolation,
        random_aspect_ratio=[0.75, 4.0 / 3.0],
        random_area=[0.08, 1.0],
        num_attempts=100,
        antialias=False)


    images = fn.flip(images, horizontal=rng)

    output = apply_policies(images, policies)

    output = fn.crop_mirror_normalize(output.gpu(),
        dtype=types.FLOAT,
        output_layout=types.NCHW,
        crop=(crop, crop),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255])


    return output, labels
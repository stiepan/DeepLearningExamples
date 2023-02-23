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

from nvidia.dali.pipeline.experimental import pipeline_def

from nvidia.dali.auto_aug import auto_augment, trivial_augment
from nvidia.dali.auto_aug import augmentations as a

no_arithm_policy = auto_augment.Policy(
    "ImageNet", 11, {
        "shear_x": a.shear_x.augmentation((0, 0.3), True),
        "shear_y": a.shear_y.augmentation((0, 0.3), True),
        "translate_x": a.translate_x.augmentation((0, 0.45), True),
        "translate_y": a.translate_y.augmentation((0, 0.45), True),
        "rotate": a.rotate.augmentation((0, 30), True),
        "brightness": a.brightness.augmentation((0.1, 1.9), False, None),
        "contrast": a.contrast.augmentation((0.1, 1.9), False, None),
        "color": a.color.augmentation((0.1, 1.9), False, None),
        "posterize": a.posterize.augmentation((0, 4), False, a.poster_mask_uint8),
        "equalize": a.equalize,
    }, [
        [("equalize", 0.8, 1), ('shear_y', 0.8, 4)],
        [('color', 0.4, 9), ('equalize', 0.6, 3)],
        [('color', 0.4, 1), ('rotate', 0.6, 8)],
        [('translate_x', 0.8, 3), ('equalize', 0.4, 7)],
        [('translate_x', 0.4, 2), ('translate_x', 0.6, 2)],
        [('color', 0.2, 0), ('equalize', 0.8, 8)],
        [('equalize', 0.4, 8), ('shear_x', 0.8, 3)],
        [('shear_x', 0.2, 9), ('rotate', 0.6, 8)],
        [('color', 0.6, 1), ('equalize', 1.0, 2)],
        [('brightness', 0.4, 9), ('rotate', 0.6, 0)],
        [('equalize', 1.0, 9), ('shear_y', 0.6, 3)],
        [('color', 0.4, 7), ('equalize', 0.6, 0)],
        [('posterize', 0.4, 6), ('contrast', 0.4, 7)],
        [('translate_x', 0.6, 8), ('color', 0.6, 9)],
        [('translate_x', 0.2, 4), ('rotate', 0.8, 9)],
        [('rotate', 1.0, 7), ('translate_y', 0.8, 9)],
        [('shear_x', 0.0, 0), ('translate_x', 0.8, 4)],
        [('shear_y', 0.8, 0), ('color', 0.6, 4)],
        [('color', 1.0, 0), ('rotate', 0.6, 2)],
        [('equalize', 0.8, 4)],
        [('equalize', 1.0, 4), ('contrast', 0.6, 2)],
        [('shear_y', 0.4, 7), ('shear_x', 0.6, 7)],
        [('posterize', 0.8, 2), ('translate_x', 0.6, 10)],
        [('translate_x', 0.6, 8), ('equalize', 0.6, 1)],
        [('color', 0.8, 6), ('rotate', 0.4, 5)],
    ])


@pipeline_def(enable_conditionals=True)
def aa_pipe(data_dir, interpolation, crop, dali_cpu=False, rank=0, world_size=1, cpu_gpu=0):
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
    shapes = fn.peek_image_shape(jpegs)

    images = fn.random_resized_crop(
        images,
        size=[crop, crop],
        interp_type=interpolation,
        random_aspect_ratio=[0.75, 4.0 / 3.0],
        random_area=[0.08, 1.0],
        num_attempts=100,
        antialias=False)


    images = fn.flip(images, horizontal=rng)

    output = auto_augment.apply_auto_augment(no_arithm_policy, images, shapes=shapes)

    output = fn.crop_mirror_normalize(output,
        dtype=types.FLOAT,
        output_layout=types.NCHW,
        crop=(crop, crop),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255])


    return output, labels

@pipeline_def(enable_conditionals=True)
def ta_pipe(data_dir, interpolation, crop, dali_cpu=False, rank=0, world_size=1, cpu_gpu=0):
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

    output = trivial_augment.trivial_augment_wide(images)

    output = fn.crop_mirror_normalize(output.gpu(),
        dtype=types.FLOAT,
        output_layout=types.NCHW,
        crop=(crop, crop),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255])


    return output, labels
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


@pipeline_def(enable_conditionals=True)
def aa_pipe(data_dir, interpolation, crop, dali_cpu=False, rank=0, world_size=1, cpu_gpu=0):
    print(f"Building DALI with AutoAugment, {auto_augment.auto_augment_image_net}")
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

    output = auto_augment.apply_auto_augment(auto_augment.auto_augment_image_net, images, shapes=shapes)

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
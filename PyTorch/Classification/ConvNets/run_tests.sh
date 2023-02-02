#!/bin/bash -x
# Variants:

python ./multiproc.py --nproc_per_node 8 ./main.py --arch efficientnet-b0 --label-smoothing 0.1 --data-backend dali-gpu --augmentation autoaugment  --batch-size 64 --epochs 4 --workspace /klecki/raport2/ --raport-file raport-aa-gpu.json /imgnet

python ./multiproc.py --nproc_per_node 8 ./main.py --arch efficientnet-b0 --label-smoothing 0.1 --data-backend pytorch --augmentation autoaugment --batch-size 64 --epochs 4 --workspace /klecki/raport2/ --raport-file raport-pytorch-aa.json /imgnet

python ./multiproc.py --nproc_per_node 8 ./main.py --arch efficientnet-b0 --label-smoothing 0.1 --data-backend pytorch --batch-size 64 --epochs 4 --workspace /klecki/raport2/ --raport-file raport-pytorch.json /imgnet

python ./multiproc.py --nproc_per_node 8 ./main.py --arch efficientnet-b0 --label-smoothing 0.1 --data-backend dali-cpu --batch-size 64 --epochs 4 --workspace /klecki/raport2/ --raport-file raport-cpu.json /imgnet

python ./multiproc.py --nproc_per_node 8 ./main.py --arch efficientnet-b0 --label-smoothing 0.1 --data-backend dali-gpu --batch-size 64 --epochs 4 --workspace /klecki/raport2/ --raport-file raport-gpu.json /imgnet

python ./multiproc.py --nproc_per_node 8 ./main.py --arch efficientnet-b0 --label-smoothing 0.1 --data-backend dali-cpu --augmentation autoaugment  --batch-size 64 --epochs 4 --workspace /klecki/raport2/ --raport-file raport-aa-cpu.json /imgnet

# PYTORCH IS DEAD :VVV

# Same as above with:


# python ./multiproc.py --nproc_per_node 8 ./main.py --arch efficientnet-b0 --label-smoothing 0.1 --data-backend dali-cpu --batch-size 64 --training-only  --epochs 4 --prof 100 --workspace /klecki/raport2/ --raport-file raport-cpu-bm.json /imgnet

# python ./multiproc.py --nproc_per_node 8 ./main.py --arch efficientnet-b0 --label-smoothing 0.1 --data-backend dali-gpu --batch-size 64 --training-only  --epochs 4 --prof 100 --workspace /klecki/raport2/ --raport-file raport-gpu-bm.json /imgnet

# python ./multiproc.py --nproc_per_node 8 ./main.py --arch efficientnet-b0 --label-smoothing 0.1 --data-backend dali-cpu --augmentation autoaugment  --batch-size 64 --training-only  --epochs 4 --prof 100 --workspace /klecki/raport2/ --raport-file raport-aa-cpu-bm.json /imgnet

# python ./multiproc.py --nproc_per_node 8 ./main.py --arch efficientnet-b0 --label-smoothing 0.1 --data-backend dali-gpu --augmentation autoaugment  --batch-size 64 --training-only  --epochs 4 --prof 100 --workspace /klecki/raport2/ --raport-file raport-aa-gpu-bm.json /imgnet

# python ./multiproc.py --nproc_per_node 8 ./main.py --arch efficientnet-b0 --label-smoothing 0.1 --data-backend pytorch --batch-size 64 --training-only  --epochs 4 --prof 100 --workspace /klecki/raport2/ --raport-file raport-pytorch-bm.json /imgnet

# python ./multiproc.py --nproc_per_node 8 ./main.py --arch efficientnet-b0 --label-smoothing 0.1 --data-backend pytorch --augmentation autoaugment --batch-size 64 --training-only  --epochs 4 --prof 100 --workspace /klecki/raport2/ --raport-file raport-pytorch-aa-bm.json /imgnet

# EDM-FDM
torchrun --standalone --nproc_per_node=4 train.py --outdir=training-output \
    --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp \
    --precond=fdm_edm --warmup_ite=200 --fp16=True

# VP-FDM
#torchrun --standalone --nproc_per_node=8 train.py --outdir=training-output \
#    --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp --cres=1,2,2,2 \
#    --precond=fdm_vp --warmup_ite=400

# VE-FDM
#torchrun --standalone --nproc_per_node=8 train.py --outdir=training-output \
#    --data=datasets/cifar10-32x32.zip --cond=1 --arch=ncsnpp --cres=1,2,2,2 \
#    --precond=fdm_ve --warmup_ite=400 
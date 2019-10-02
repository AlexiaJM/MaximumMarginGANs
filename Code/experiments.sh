#!/bin/bash

cd /home/jolicoea/my_projects/MaxMargin/Code

### Data export
bash startup_tmp.sh dir1="CIFAR10" dir2="Meow_64x64" dir3="Meow_256x256"





## CIFAR-10 (.50,.99 adam)

python GAN.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True  --l1_margin
bash fid_script.sh 10 "adam0/.50 HingeGAN CIFAR-10 lr .0002 linf squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --l1_margin
bash fid_script.sh 10 "WGAN CIFAR-10 lr .0002 linf squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True --penalty-type 'hinge' --l1_margin
bash fid_script.sh 10 "HingeGAN CIFAR-10 lr .0002 linf hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --penalty-type 'hinge' --l1_margin
bash fid_script.sh 10 "WGAN CIFAR-10 lr .0002 linf hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output


python GAN.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True  --l1_margin_smoothmax --smoothmax 1
bash fid_script.sh 10 "HingeGAN CIFAR-10 lr .0002 linf-smooth1 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --l1_margin_smoothmax --smoothmax 1
bash fid_script.sh 10 "WGAN CIFAR-10 lr .0002 linf-smooth1 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True --penalty-type 'hinge' --l1_margin_smoothmax --smoothmax 1
bash fid_script.sh 10 "HingeGAN CIFAR-10 lr .0002 linf-smooth1 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --penalty-type 'hinge' --l1_margin_smoothmax --smoothmax 1
bash fid_script.sh 10 "WGAN CIFAR-10 lr .0002 linf-smooth1 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output


python GAN.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True
bash fid_script.sh 10 "HingeGAN CIFAR-10 lr .0002 l2 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True
bash fid_script.sh 10 "WGAN CIFAR-10 lr .0002 l2 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True --penalty-type 'hinge'
bash fid_script.sh 10 "HingeGAN CIFAR-10 lr .0002 l2 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --penalty-type 'hinge'
bash fid_script.sh 10 "WGAN CIFAR-10 lr .0002 l2 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output


python GAN.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True --linf_margin
bash fid_script.sh 10 "HingeGAN CIFAR-10 lr .0002 l1 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --linf_margin
bash fid_script.sh 10 "WGAN CIFAR-10 lr .0002 l1 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True --penalty-type 'hinge' --linf_margin
bash fid_script.sh 10 "HingeGAN CIFAR-10 lr .0002 l1 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --penalty-type 'hinge' --linf_margin
bash fid_script.sh 10 "WGAN CIFAR-10 lr .0002 l1 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output





## CIFAR-10 (0,.9 adam)

python GAN.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --beta1 0 --beta2 .90 --grad_penalty True  --l1_margin
bash fid_script.sh 10 "adam0/.50 HingeGAN CIFAR-10 lr .0002 linf squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --beta1 0 --beta2 .90 --l1_margin
bash fid_script.sh 10 "WGAN CIFAR-10 lr .0002 linf squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --beta1 0 --beta2 .90 --grad_penalty True --penalty-type 'hinge' --l1_margin
bash fid_script.sh 10 "HingeGAN CIFAR-10 lr .0002 linf hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --beta1 0 --beta2 .90 --penalty-type 'hinge' --l1_margin
bash fid_script.sh 10 "WGAN CIFAR-10 lr .0002 linf hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output


python GAN.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --beta1 0 --beta2 .90 --grad_penalty True  --l1_margin_smoothmax --smoothmax 1
bash fid_script.sh 10 "HingeGAN CIFAR-10 lr .0002 linf-smooth1 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --beta1 0 --beta2 .90 --l1_margin_smoothmax --smoothmax 1
bash fid_script.sh 10 "WGAN CIFAR-10 lr .0002 linf-smooth1 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --beta1 0 --beta2 .90 --grad_penalty True --penalty-type 'hinge' --l1_margin_smoothmax --smoothmax 1
bash fid_script.sh 10 "HingeGAN CIFAR-10 lr .0002 linf-smooth1 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --beta1 0 --beta2 .90 --penalty-type 'hinge' --l1_margin_smoothmax --smoothmax 1
bash fid_script.sh 10 "WGAN CIFAR-10 lr .0002 linf-smooth1 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output


python GAN.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --beta1 0 --beta2 .90 --grad_penalty True
bash fid_script.sh 10 "HingeGAN CIFAR-10 lr .0002 l2 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --beta1 0 --beta2 .90
bash fid_script.sh 10 "WGAN CIFAR-10 lr .0002 l2 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --beta1 0 --beta2 .90 --grad_penalty True --penalty-type 'hinge'
bash fid_script.sh 10 "HingeGAN CIFAR-10 lr .0002 l2 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --beta1 0 --beta2 .90 --penalty-type 'hinge'
bash fid_script.sh 10 "WGAN CIFAR-10 lr .0002 l2 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output


python GAN.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --beta1 0 --beta2 .90 --grad_penalty True --linf_margin
bash fid_script.sh 10 "HingeGAN CIFAR-10 lr .0002 l1 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --beta1 0 --beta2 .90 --linf_margin
bash fid_script.sh 10 "WGAN CIFAR-10 lr .0002 l1 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --beta1 0 --beta2 .90 --grad_penalty True --penalty-type 'hinge' --linf_margin
bash fid_script.sh 10 "HingeGAN CIFAR-10 lr .0002 l1 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --beta1 0 --beta2 .90 --penalty-type 'hinge' --linf_margin
bash fid_script.sh 10 "WGAN CIFAR-10 lr .0002 l1 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output




## Meow 64

python GAN.py --loss_D 3 --input_folder '/Datasets/Meow_64x64' --image_size 64 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --G_h_size 64 --D_h_size 64 --grad_penalty True
bash fid_script.sh 10 "Hinge Meow-64 lr .0002 l-2 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/CAT_fid_stats64.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 3 --input_folder '/Datasets/Meow_64x64' --image_size 64 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --G_h_size 64 --D_h_size 64 --grad_penalty True --l1_margin --penalty-type 'hinge'
bash fid_script.sh 10 "Hinge Meow-64 lr .0002 l-inf hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/CAT_fid_stats64.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 3 --input_folder '/Datasets/Meow_64x64' --image_size 64 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --G_h_size 64 --D_h_size 64 --grad_penalty True --penalty-type 'hinge'
bash fid_script.sh 10 "HingeGAN Meow-64 lr .0002 l-2 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/CAT_fid_stats64.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output


python GAN.py --loss_D 4 --input_folder '/Datasets/Meow_64x64' --image_size 64 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --G_h_size 64 --D_h_size 64 --grad_penalty True
bash fid_script.sh 10 "WGAN Meow-64 lr .0002 l-2 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/CAT_fid_stats64.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --input_folder '/Datasets/Meow_64x64' --image_size 64 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --G_h_size 64 --D_h_size 64 --grad_penalty True --l1_margin --penalty-type 'hinge'
bash fid_script.sh 10 "WGAN Meow-64 lr .0002 l-inf hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/CAT_fid_stats64.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --input_folder '/Datasets/Meow_64x64' --image_size 64 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --G_h_size 64 --D_h_size 64 --grad_penalty True --penalty-type 'hinge'
bash fid_script.sh 10 "WGAN Meow-64 lr .0002 l-2 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/CAT_fid_stats64.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output




## CIFAR-10 resnet

python GAN.py --loss_D 3 --CIFAR10 True --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --G_h_size 64 --D_h_size 64 --grad_penalty True --arch 2
bash fid_script.sh 10 "Hinge CIFAR10-resnet lr .0002 l-2 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 3 --CIFAR10 True --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --G_h_size 64 --D_h_size 64 --grad_penalty True  --arch 2 --l1_margin --penalty-type 'hinge'
bash fid_script.sh 10 "Hinge CIFAR10-resnet lr .0002 l-inf hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 3 --CIFAR10 True --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --G_h_size 64 --D_h_size 64 --grad_penalty True --arch 2 --penalty-type 'hinge'
bash fid_script.sh 10 "HingeGAN CIFAR10-resnet lr .0002 l-2 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output


python GAN.py --loss_D 4 --CIFAR10 True --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --G_h_size 64 --D_h_size 64 --grad_penalty True --arch 2
bash fid_script.sh 10 "WGAN CIFAR10-resnet lr .0002 l-2 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --CIFAR10 True --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --G_h_size 64 --D_h_size 64 --grad_penalty True --arch 2 --l1_margin --penalty-type 'hinge'
bash fid_script.sh 10 "WGAN CIFAR10-resnet lr .0002 l-inf hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --CIFAR10 True --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --G_h_size 64 --D_h_size 64 --grad_penalty True --arch 2 --penalty-type 'hinge'
bash fid_script.sh 10 "WGAN CIFAR10-resnet lr .0002 l-2 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output


python GAN.py --loss_D 3 --CIFAR10 True --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --G_h_size 64 --D_h_size 64 --grad_penalty True --arch 2
bash fid_script.sh 10 "Hinge CIFAR10-resnet lr .0002 l-2 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 3 --CIFAR10 True --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --G_h_size 64 --D_h_size 64 --grad_penalty True --arch 2 --l1_margin --penalty-type 'hinge'
bash fid_script.sh 10 "Hinge CIFAR10-resnet lr .0002 l-inf hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 3 --CIFAR10 True --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --G_h_size 64 --D_h_size 64 --grad_penalty True --arch 2 --penalty-type 'hinge'
bash fid_script.sh 10 "HingeGAN CIFAR10-resnet lr .0002 l-2 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output


python GAN.py --loss_D 4 --CIFAR10 True --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --G_h_size 64 --D_h_size 64 --grad_penalty True --arch 2
bash fid_script.sh 10 "WGAN CIFAR10-resnet lr .0002 l-2 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --CIFAR10 True --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --G_h_size 64 --D_h_size 64 --grad_penalty True --arch 2 --l1_margin --penalty-type 'hinge'
bash fid_script.sh 10 "WGAN CIFAR10-resnet lr .0002 l-inf hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 4 --CIFAR10 True --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --G_h_size 64 --D_h_size 64 --grad_penalty True --arch 2 --penalty-type 'hinge'
bash fid_script.sh 10 "WGAN CIFAR10-resnet lr .0002 l-2 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output




## RaGANs

python GAN.py --loss_D 13 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True --l1_margin
bash fid_script.sh 10 "RaHingeGAN CIFAR-10 lr .0002 l-inf squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output
python GAN.py --loss_D 13 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True --l1_margin --penalty-type 'hinge'
bash fid_script.sh 10 "RaHingeGAN CIFAR-10 lr .0002 l-inf hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 13 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True --l1_margin_smoothmax --smoothmax 1
bash fid_script.sh 10 "RaHingeGAN CIFAR-10 lr .0002 l-inf squared-1 penalty smoothmax 1" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output
python GAN.py --loss_D 13 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True --l1_margin_smoothmax --smoothmax 1 --penalty-type 'hinge'
bash fid_script.sh 10 "RaHingeGAN CIFAR-10 lr .0002 l-inf hinge penalty smoothmax 1" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output


python GAN.py --loss_D 13 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True
bash fid_script.sh 10 "RaHingeGAN CIFAR-10 lr .0002 l-2 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output
python GAN.py --loss_D 13 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True --penalty-type 'hinge'
bash fid_script.sh 10 "RaHingeGAN CIFAR-10 lr .0002 l-2 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 13 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True --linf_margin
bash fid_script.sh 10 "RaHingeGAN CIFAR-10 lr .0002 l-1 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output
python GAN.py --loss_D 13 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True --penalty-type 'hinge' --linf_margin
bash fid_script.sh 10 "RaHingeGAN CIFAR-10 lr .0002 l-1 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output




## RpGANs

python GAN.py --loss_D 33 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True --l1_margin
bash fid_script.sh 10 "RpHingeGAN CIFAR-10 lr .0002 l-inf squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output
python GAN.py --loss_D 33 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True --l1_margin --penalty-type 'hinge'
bash fid_script.sh 10 "RpHingeGAN CIFAR-10 lr .0002 l-inf hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 33 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True --l1_margin_smoothmax --smoothmax 1
bash fid_script.sh 10 "RpHingeGAN CIFAR-10 lr .0002 l-inf squared-1 penalty smoothmax 1" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output
python GAN.py --loss_D 33 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True --l1_margin_smoothmax --smoothmax 1 --penalty-type 'hinge'
bash fid_script.sh 10 "RpHingeGAN CIFAR-10 lr .0002 l-inf hinge penalty smoothmax 1" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output


python GAN.py --loss_D 33 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True
bash fid_script.sh 10 "RpHingeGAN CIFAR-10 lr .0002 l-2 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output
python GAN.py --loss_D 33 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True --penalty-type 'hinge'
bash fid_script.sh 10 "RpHingeGAN CIFAR-10 lr .0002 l-2 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output

python GAN.py --loss_D 33 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True --linf_margin
bash fid_script.sh 10 "RpHingeGAN CIFAR-10 lr .0002 l-1 squared-1 penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output
python GAN.py --loss_D 33 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --n_iter 100001 --gen_every 10000 --print_every 5000 --gen_extra_images 50000 --CIFAR10 True --grad_penalty True --penalty-type 'hinge' --linf_margin
bash fid_script.sh 10 "RpHingeGAN CIFAR-10 lr .0002 l-1 hinge penalty" 10000 "$SLURM_TMPDIR/fid_stats/fid_stats_cifar10_train.npz"
rsync -r $SLURM_TMPDIR/Output/GANlosses /scratch/jolicoea/Output
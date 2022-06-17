CUDA_VISIBLE_DEVICES=0 python train.py --name cifar10_resnet18_esam --log wandb --mode esam --beta 0.5 --gamma 0.5 --learning_rate 0.05 --weight_decay 1e-3  --rho 0.05 --batch_size 256 --arch resnet18 --dataset cifar10 --epochs 250
CUDA_VISIBLE_DEVICES=1 python train.py --name cifar10_resnet18_sam --log wandb --mode esam --beta 1.0 --gamma 1.0 --learning_rate 0.05 --weight_decay 1e-3  --rho 0.05 --batch_size 256 --arch resnet18 --dataset cifar10 --epochs 250
CUDA_VISIBLE_DEVICES=2 python train.py --name cifar10_resnet18_sgd --log wandb --mode sgd --learning_rate 0.05 --weight_decay 1e-3 --batch_size 256 --arch resnet18 --dataset cifar10 --epochs 250

#to use the new varying_gamma method, change the code in train.py 48th line to ESAM_vary class, and then run the code same as esam.


#CUDA_VISIBLE_DEVICES=0 python train.py --name cifar10_wresnet28_esam --log wandb --mode esam --beta 0.5 --gamma 0.5  --learning_rate 0.05 --weight_decay 1e-3  --rho 0.1 --batch_size 256 --arch wideresnet18 --dataset cifar10 --epochs 250
#CUDA_VISIBLE_DEVICES=2 python train.py --name cifar10_wresnet28_sam --log wandb --mode esam --beta 1.0 --gamma 1.0 --learning_rate 0.05 --weight_decay 1e-3  --rho 0.1 --batch_size 256 --arch wideresnet18 --dataset cifar10 --epochs 250
#CUDA_VISIBLE_DEVICES=3 python train.py --name cifar10_wresnet28_sgd --log wandb --mode sgd --learning_rate 0.05 --weight_decay 1e-3 --batch_size 256 --arch wideresnet18 --dataset cifar10 --epochs 250

#CUDA_VISIBLE_DEVICES=0 python train.py --name cifar10_pyrm_esam --log wandb --mode esam --beta 0.5 --gamma 0.5  --learning_rate 0.1 --weight_decay 5e-4  --rho 0.2 --batch_size 256 --arch pyrm --dataset cifar10 --epochs 350
#CUDA_VISIBLE_DEVICES=1 python train.py --name cifar10_pyrm_sam --log wandb --mode esam --learning_rate 0.1 --weight_decay 5e-4  --rho 0.2 --batch_size 256 --arch pyrm --dataset cifar10 --epochs 350
#CUDA_VISIBLE_DEVICES=2 python train.py --name cifar10_pyrm_sgd --log wandb --mode sgd --learning_rate 0.1 --weight_decay 5e-4 --batch_size 256 --arch pyrm --dataset cifar10 --epochs 350



#CUDA_VISIBLE_DEVICES=0 python train.py --name cifar100_resnet18_esam --log wandb --mode esam --beta 0.5 --gamma 0.5  --learning_rate 0.05 --weight_decay 1e-3  --rho 0.05 --batch_size 256 --arch resnet18 --dataset cifar100 --epochs 250
#CUDA_VISIBLE_DEVICES=1 python train.py --name cifar100_resnet18_sam --log wandb --mode esam --beta 1.0 --gamma 1.0 --learning_rate 0.05 --weight_decay 1e-3  --rho 0.05 --batch_size 256 --arch resnet18 --dataset cifar100 --epochs 250
#CUDA_VISIBLE_DEVICES=2 python train.py --name cifar100_resnet18_sgd --log wandb --mode sgd --learning_rate 0.05 --weight_decay 1e-3 --batch_size 256 --arch resnet18 --dataset cifar100 --epochs 250

#CUDA_VISIBLE_DEVICES=0 python train.py --name cifar100_wresnet28_esam --log wandb --mode esam --beta 0.5 --gamma 0.5  --learning_rate 0.05 --weight_decay 1e-3  --rho 0.1 --batch_size 256 --arch wideresnet18 --dataset cifar100 --epochs 250
#CUDA_VISIBLE_DEVICES=1 python train.py --name cifar100_wresnet28_sam --log wandb --mode esam --beta 1.0 --gamma 1.0 --learning_rate 0.05 --weight_decay 1e-3  --rho 0.1 --batch_size 256 --arch wideresnet18 --dataset cifar100 --epochs 250
#CUDA_VISIBLE_DEVICES=2 python train.py --name cifar100_wresnet28_sgd --log wandb --mode sgd --learning_rate 0.05 --weight_decay 1e-3 --batch_size 256 --arch wideresnet18 --dataset cifar100 --epochs 250

#CUDA_VISIBLE_DEVICES=0 python train.py --name cifar100_pyrm_esam --log wandb --mode esam --beta 0.5 --gamma 0.5  --learning_rate 0.1 --weight_decay 5e-4  --rho 0.2 --batch_size 256 --arch pyrm --dataset cifar100 --epochs 350
#CUDA_VISIBLE_DEVICES=1 python train.py --name cifar100_pyrm_sam --log wandb --mode esam --beta 1.0 --gamma 1.0 --learning_rate 0.1 --weight_decay 5e-4  --rho 0.2 --batch_size 256 --arch pyrm --dataset cifar100 --epochs 350
#CUDA_VISIBLE_DEVICES=2 python train.py --name cifar100_pyrm_sgd --log wandb --mode sgd --learning_rate 0.1 --weight_decay 5e-4 --batch_size 256 --arch pyrm --dataset cifar100 --epochs 350

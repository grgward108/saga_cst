import argparse
import os
import sys

from utils.cfg_parser import Config
from WholeGraspPose.trainer import Trainer

def count_parameters(trainer):
    """
    Count and display the number of parameters in the FullBodyGraspNet model and its components.
    """
    model = trainer.full_grasp_net

    # Total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # If the model has specific components, count their parameters
    components = {}
    for name, module in model.named_children():  # Iterate over submodules
        components[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)

    # Print details
    print(f"Total trainable parameters in FullBodyGraspNet: {total_params}")
    for name, params in components.items():
        print(f"  - {name}: {params} parameters")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GrabNet-Training')

    parser.add_argument('--work-dir', default='logs/GraspPose', type=str,
                        help='The path to the downloaded grab data')

    parser.add_argument('--gender', default=None, type=str,
                        help='The gender of dataset')

    parser.add_argument('--data_path', default = '/cluster/work/cvl/wuyan/data/GRAB-series/GrabPose_r_fullbody/data', type=str,
                        help='The path to the folder that contains grabpose data')

    parser.add_argument('--batch-size', default=64, type=int,
                        help='Training batch size')

    parser.add_argument('--n-workers', default=8, type=int,
                        help='Number of PyTorch dataloader workers')

    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Training learning rate')

    parser.add_argument('--kl-coef', default=0.5, type=float,
                        help='KL divergence coefficent for Coarsenet training')

    parser.add_argument('--use-multigpu', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='If to use multiple GPUs for training')

    parser.add_argument('--exp_name', default = None, type=str,
                        help='experiment name')
    
    parser.add_argument('--param-count', action='store_true',
                        help='If set, the script will display parameter counts and exit.')


    args = parser.parse_args()

    work_dir = os.path.join(args.work_dir, args.exp_name)

    cwd = os.getcwd()
    default_cfg_path = 'WholeGraspPose/configs/WholeGraspPose.yaml'

    cfg = {
        'batch_size': args.batch_size,
        'n_workers': args.n_workers,
        'use_multigpu': args.use_multigpu,
        'kl_coef': args.kl_coef,
        'dataset_dir': args.data_path,
        'base_dir': cwd,
        'work_dir': work_dir,
        'base_lr': args.lr,
        'best_net': None,
        'gender': args.gender,
        'exp_name': args.exp_name,
    }

    cfg = Config(default_cfg_path=default_cfg_path, **cfg)
    grabpose_trainer = Trainer(cfg=cfg)

    if args.param_count:
        count_parameters(grabpose_trainer)
        sys.exit(0)

    grabpose_trainer.fit()

    cfg = grabpose_trainer.cfg
    cfg.write_cfg(os.path.join(work_dir, 'TR%02d_%s' % (cfg.try_num, os.path.basename(default_cfg_path))))

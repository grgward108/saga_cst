import argparse
import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from PoseBridge.data.dataloader import GRAB_DataLoader
from PoseBridge.models.models import PoseBridge
import logging


# Argument parser for configuration
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='dataset/GraspMotion')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--log_step', type=int, default=100)
parser.add_argument('--save_step', type=int, default=500)
parser.add_argument('--save_dir', type=str, default='logs')
parser.add_argument('--exp_name', type=str, default='default_experiment', help='Experiment name for organizing logs')

args = parser.parse_args()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logger(logdir):
    """
    Set up the logger to write to both console and a log file.
    """
    logger = logging.getLogger("PoseBridge")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # File handler
    log_file = os.path.join(logdir, f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Training function
def train():
    # Logger and directory setup
    logdir = os.path.join('posebridge_logs', 'PoseBridge', args.exp_name, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=logdir)
    logger = setup_logger(logdir)
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Logs will be saved in: {logdir}")

    # Dataset setup
    train_dataset = GRAB_DataLoader(clip_seconds=2, clip_fps=30, split='train', normalize=True, markers_type='f15_p22')
    train_dataset.read_data(['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8'], args.dataset_dir)
    train_dataset.create_body_hand_repr(smplx_model_path='body_utils/body_models')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_dataset = GRAB_DataLoader(clip_seconds=2, clip_fps=30, split='test', normalize=True, markers_type='f15_p22')
    test_dataset.read_data(['s9', 's10'], args.dataset_dir)
    test_dataset.create_body_hand_repr(smplx_model_path='body_utils/body_models')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    n_markers = len(train_dataset.markers_ids)

    # Model and optimizer setup
    model = PoseBridge(n_markers=n_markers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.L1Loss()

    logger.info("Training started.")

    # Training loop
    for epoch in range(args.num_epoch):
        logger.info(f"Epoch {epoch+1}/{args.num_epoch}")
        model.train()
        total_loss = 0

        for step, (masked_markers, ground_truth, part_labels) in enumerate(train_loader):
            # Data to device
            masked_markers = masked_markers.to(device)
            ground_truth = ground_truth.to(device)
            part_labels = part_labels.to(device)

            # Forward pass
            predicted_markers = model(masked_markers, part_labels)

            loss_rec_body = criterion(predicted_markers, ground_truth)
            loss_rec_body.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss_rec_body.item()

            if (step + 1) % args.log_step == 0:
                avg_loss = total_loss / (step + 1)
                logger.info(f"[Epoch {epoch+1}/{args.num_epoch}] Step {step+1}/{len(train_loader)} | Loss: {avg_loss:.6f}")
                writer.add_scalar('train/loss_rec_body', avg_loss, epoch * len(train_loader) + step)

        scheduler.step()

        # Evaluation on test set
        if (epoch + 1) % 10 == 0 or epoch == args.num_epoch - 1:
            model.eval()
            total_test_loss = 0
            with torch.no_grad():
                for step, (masked_markers, ground_truth, part_labels) in enumerate(test_loader):
                    masked_markers = masked_markers.to(device)
                    ground_truth = ground_truth.to(device)
                    part_labels = part_labels.to(device)

                    # Forward pass
                    predicted_markers = model(masked_markers, part_labels)
                    loss_rec_body = criterion(predicted_markers, ground_truth)
                    total_test_loss += loss_rec_body.item()

                avg_test_loss = total_test_loss / len(test_loader)
                logger.info(f"[Epoch {epoch+1}/{args.num_epoch}] Test Loss: {avg_test_loss:.6f}")
                writer.add_scalar('test/loss_rec_body', avg_test_loss, epoch + 1)

        # Save model checkpoint
        if (epoch + 1) % args.save_step == 0 or epoch == args.num_epoch - 1:
            save_path = os.path.join(logdir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, save_path)
            logger.info(f"Model checkpoint saved at {save_path}")

    logger.info("Training completed.")


if __name__ == "__main__":
    train()

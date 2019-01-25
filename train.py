from __future__ import division

import argparse
from datetime import datetime

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.parse_config import *
from utils.utils import *


def main(opt):
    cuda = torch.cuda.is_available() and opt.use_cuda

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)

    today = datetime.today().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=os.path.join(opt.log_dir, today))

    # classes = load_classes(opt.class_path)

    # Get data configuration
    data_config = parse_data_config(opt.data_config_path)
    train_path = data_config["train"]
    val_path = data_config["valid"]

    # Get hyper parameters
    # hyperparams = parse_model_config(opt.model_config_path)[0]
    # learning_rate = float(hyperparams["learning_rate"])
    # momentum = float(hyperparams["momentum"])
    # decay = float(hyperparams["decay"])
    # burn_in = int(hyperparams["burn_in"])

    # Initiate model
    model = Darknet(opt.model_config_path)
    model = torch.nn.DataParallel(model)
    # model.load_weights(opt.weights_path)
    model.apply(weights_init_normal)

    if cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=opt.wd)

    # Load previous checkpoint if start-epoch > 0
    if opt.start_epochs > 0:
        assert opt.epochs > opt.start_epochs

        checkpoint = torch.load(os.path.join(opt.checkpoint_dir, f'{opt.start_epochs}.weights'))

        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # opt_state_dict = torch.load(os.path.join(
        #    opt.checkpoint_dir, f'{opt.start_epochs}.opt.weights'))

        # model.module.load_weights(f"{opt.checkpoint_dir}/{opt.start_epochs}.weights")

    model.train()

    # Get dataloaders
    dataloader = torch.utils.data.DataLoader(
        ListDataset(train_path), batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu
    )
    val_dataloader = torch.utils.data.DataLoader(
        ListDataset(val_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for epoch in range(opt.start_epochs, opt.epochs):
        model.train()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            imgs = Variable(imgs.type(Tensor))
            targets = Variable(targets.type(Tensor), requires_grad=False)

            optimizer.zero_grad()

            loss = model(imgs, targets)

            loss.mean().backward()
            optimizer.step()

            print(f'[Epoch {1 + epoch}/{opt.epochs}, Batch {1 + batch_i}/{len(dataloader)}] '
                  #     f'[Losses: x {model.module.losses["x"]:.5f}, y {model.module.losses["y"]:.5f}, '
                  #     f'w {model.module.losses["w"]:.5f}, h {model.module.losses["h"]:.5f}, '
                  #     f'conf {model.module.losses["conf"]:.5f}, cls {model.module.losses["cls"]:.5f}, '
                  f'total {loss.mean().item():.5f} ')
            #    f'total {loss.item():.5f}, recall: {model.module.losses["recall"]:.5f}, '
            #    f'precision: {model.module.losses["precision"]:.5f}]')

            # Save results to TensorBoard
            curr_step = epoch * len(dataloader) + batch_i

            # writer.add_scalar('train/loss_x', model.module.losses["x"], curr_step)
            # writer.add_scalar('train/loss_y', model.module.losses["y"], curr_step)
            # writer.add_scalar('train/loss_w', model.module.losses["w"], curr_step)
            # writer.add_scalar('train/loss_h', model.module.losses["h"], curr_step)
            # writer.add_scalar('train/loss_conf', model.module.losses["conf"], curr_step)
            # writer.add_scalar('train/loss_cls', model.module.losses["cls"], curr_step)
            writer.add_scalar('train/loss_total', loss.mean().item(), curr_step)
            # writer.add_scalar('train/precision', model.module.losses["precision"], curr_step)
            # writer.add_scalar('train/recall', model.module.losses["recall"], curr_step)

            model.module.seen += imgs.size(0)

        # Validation
        with torch.no_grad():
            model.eval()

            for val_batch_i, (_, imgs, targets) in enumerate(val_dataloader):
                imgs = Variable(imgs.type(Tensor))
                targets = Variable(targets.type(Tensor), requires_grad=False)

                val_loss = model(imgs, targets)

                print(f'[Validation, Batch {1 + val_batch_i}/{len(val_dataloader)}] '
                      #                      f'[Losses: x {model.losses["x"]:.5f}, y {model.losses["y"]:.5f}, '
                      #                      f'w {model.losses["w"]:.5f}, h {model.losses["h"]:.5f}, '
                      #                      f'conf {model.losses["conf"]:.5f}, cls {model.losses["cls"]:.5f}, '
                      #                      f'total {val_loss.item():.5f}, recall: {model.losses["recall"]:.5f}, '
                      f'total {val_loss.mean().item():.5f}')
                #                      f'precision: {model.losses["precision"]:.5f}]')

                # Save results to TensorBoard
                curr_step = epoch * len(val_dataloader) + val_batch_i

                #                writer.add_scalar('val/loss_x', model.module.losses["x"], curr_step)
                #                writer.add_scalar('val/loss_y', model.module.losses["y"], curr_step)
                #                writer.add_scalar('val/loss_w', model.module.losses["w"], curr_step)
                #                writer.add_scalar('val/loss_h', model.module.losses["h"], curr_step)
                #                writer.add_scalar('val/loss_conf', model.module.losses["conf"], curr_step)
                #                writer.add_scalar('val/loss_cls', model.module.losses["cls"], curr_step)
                writer.add_scalar('val/loss_total', val_loss.mean().item(), curr_step)
        #               writer.add_scalar('val/precision', model.module.losses["precision"], curr_step)
        #               writer.add_scalar('val/recall', model.module.losses["recall"], curr_step)

        if epoch % opt.checkpoint_interval == 0:
            # model.module.save_weights(f"{opt.checkpoint_dir}/{epoch}.weights")
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(opt.checkpoint_dir, f'{epoch + 1}.weights'))

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train YOLOv3',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Numeric arguments
    parser.add_argument("--epochs", type=int, default=30,
                        help="number of epochs")
    parser.add_argument("--start-epochs", type=int, default=0,
                        help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="size of each image batch")
    parser.add_argument("--conf_thres", type=float, default=0.8,
                        help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4,
                        help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=0,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416,
                        help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1,
                        help="interval between saving model weights")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate of the Adam optimizer")
    parser.add_argument("--wd", type=float, default=1e-4,
                        help="Weight decay of the Adam optimizer")

    # Path arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="directory where model checkpoints are saved")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="directory where TensorBoard logs are saved")

    # File path arguments
    parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg",
                        help="path to model config file")
    parser.add_argument("--data_config_path", type=str, default="config/coco.data",
                        help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights",
                        help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names",
                        help="path to class label file")

    # Flag arguments
    parser.add_argument("--use_cuda", type=bool, default=True,
                        help="whether to use cuda if available")

    args = parser.parse_args()

    main(args)

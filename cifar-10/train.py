import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from dataset import CUB
from network import ConvNet
import tqdm 
import torchvision.transforms as transforms
import torch.optim as optim
from util import evaluate, AverageMeter
import argparse
from torch.utils.tensorboard  import SummaryWriter
import torchvision.datasets as datasets
from PIL import Image
from torchvision import models

def validate(epoch, model, val_loader, writer):
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    iteration = len(val_loader) * epoch
    for imgs, labels in tqdm.tqdm(val_loader):
        if torch.cuda.is_available():
            imgs = imgs.to("cuda")
        bsz = labels.shape[0]
        output = model(imgs)
        # print(output)
        if torch.cuda.is_available():
            output = output.cpu()
        # update metric
        acc1, acc5 = evaluate(output, labels, topk=(1, 5))
        top1.update(acc1.item(), bsz)
        top5.update(acc5.item(), bsz)
        iteration += 1

    # ----------TODO------------
    # draw accuracy curve!
    # ----------TODO------------
    if iteration % 10 == 0:
        writer.add_scalar('valid/top1', top1.avg, iteration)
        writer.add_scalar('valid/top5', top5.avg, iteration)

    print(' Val Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' Val Acc@5 {top5.avg:.3f}'.format(top5=top5))
    return 

def train(epoch, model, optimizer, train_loader, writer):
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    iteration = len(train_loader) * epoch
    CEloss = torch.nn.CrossEntropyLoss()
    for imgs, labels in tqdm.tqdm(train_loader):
        bsz = labels.shape[0]
        
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        output, _ = model(imgs)
        loss = CEloss(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = evaluate(output, labels, topk=(1, 5))
        top1.update(acc1.item(), bsz)
        top5.update(acc5.item(), bsz)

        loss.backward()
        optimizer.step()

        iteration += 1
        if iteration % 50 == 0:
            # ----------TODO------------
            # draw loss curve and accuracy curve!
            # ----------TODO------------
            writer.add_scalar('train/loss', loss.item(), iteration)
            writer.add_scalar('train/top1', top1.avg, iteration)
            writer.add_scalar('train/top5', top5.avg, iteration)

    print(' Epoch: %d'%(epoch))
    print(' Train Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' Train Acc@5 {top5.avg:.3f}'.format(top5=top5))
    return 

def run(args):
    save_folder = os.path.join('../experiments',  args.exp_name)
    ckpt_folder = os.path.join(save_folder,  'ckpt')
    log_folder = os.path.join(save_folder,  'log')
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=log_folder)

    # define dataset and dataloader
    # train_dataset = CIFAR10()
    # val_dataset = CIFAR10(train=False)
    
    root = "/home/xbj/lable_noising/release"
    train_transform=transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.RandomCrop((384, 384)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    test_transform = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.CenterCrop((384, 384)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    train_dataset = CUB(root, is_train=True, transform = train_transform)#, target_transform=target_transform)
    val_dataset = CUB(root, is_train=False, transform = test_transform)#, target_transform=target_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
         val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=2)
    
    # define network 
    # model = ConvNet(num_class=200)
    # if torch.cuda.is_available():
    #     model = model.to("cuda")
    model = models.inception_v3(num_classes = 200).to("cuda")#, pretrained=True)
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print("Prepared")
    if args.cont:
        # load latest checkpoint
        ckpt_lst = os.listdir(ckpt_folder)
        ckpt_lst.sort(key=lambda x: int(x.split('_')[-1]))
        read_path = os.path.join(ckpt_folder, ckpt_lst[-1])
        print('load checkpoint from %s'%(read_path))
        checkpoint = torch.load(read_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    
    for epoch in range(start_epoch, args.total_epoch):
        train(epoch, model, optimizer, train_loader, writer)
        
        if epoch % args.save_freq == 0:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(ckpt_folder, 'ckpt_epoch_%s'%(str(epoch)))
            torch.save(state, save_file)

            with torch.no_grad():
                validate(epoch, model, val_loader, writer)
    
    writer.close()
    return 

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_name', '-e', type=str, required=True, help="The checkpoints and logs will be save in ./checkpoint/$EXP_NAME")
    arg_parser.add_argument('--lr', '-l', type=float, default=2e-4, help="Learning rate")
    arg_parser.add_argument('--save_freq', '-s', type=int, default=5, help="frequency of saving model")
    arg_parser.add_argument('--total_epoch', '-t', type=int, default=100, help="total epoch number for training")
    arg_parser.add_argument('--cont', '-c', action='store_true', help="whether to load saved checkpoints from $EXP_NAME and continue training")
    arg_parser.add_argument('--batchsize', '-b', type=int, default=128, help="batch size")
    args = arg_parser.parse_args()

    run(args)
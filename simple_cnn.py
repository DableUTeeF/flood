import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torch
from sklearn.metrics import f1_score
from torchvision import transforms
import timm
from tensorflow.keras.utils import Progbar
from torch.optim.lr_scheduler import MultiStepLR
from torch import nn
import numpy as np


cats = ['สายไฟ', 'น้ำท่วม', 'ป้าย', 'ทางเท้า', 'ต้นไม้', 'ความสะอาด', 'กีดขวาง', 'ถนน', 'สัตว์จรจัด', 'ความปลอดภัย', 'ป้ายจราจร', 'สะพาน', 'การเดินทาง', 'คนจรจัด', 'เสนอแนะ',
        'สอบถาม', 'ท่อระบายน้ำ', 'คลอง', 'แสงสว่าง', 'เสียงรบกวน', 'ร้องเรียน', 'จราจร', 'PM2.5', 'ห้องน้ำ']
dst = '/media/palm/Data/traffy_data/images'


class args:
    outdir = '/media/palm/Data/traffy_data/cp'
    traindir = '/media/palm/BiggerData/parasites/parasites2/train'
    valdir = '/media/palm/BiggerData/parasites/parasites2/val'
    model = 'mobilenetv3_small_075'
    imsize = 224
    batch_size = 64
    workers = 8
    val_batch_size = 64
    num_epochs = 30


def pairing(row):
    if row['saved']:
        if row['type'] != '' and len(os.path.basename(row['photo'])) > 5:
            tps = row['type'].split(',')
            cat = [cats.index(x) for x in tps]
            return cat
    return None


class CSVDataSet(Dataset):
    def __init__(self, csv='csvs/teamchadchart.csv', transform=None):
        self.data = []
        self.df = pd.read_csv(csv)
        self.transform = transform
        self.df.fillna('', inplace=True)
        saved = []
        for i in range(100):
            dff = pd.read_csv(f'csvs/saved_{i:04d}.csv')
            saved.extend(dff['saved'])
        self.df['saved'] = saved
        self.df['cats'] = self.df.apply(pairing, axis=1)
        self.df = self.df[saved]
        self.df = self.df[self.df.apply(lambda row: row['cats'] is not None, axis=1)]
        self.df = self.df[['photo', 'cats']]
        self.images = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = Image.open(os.path.join(dst, os.path.basename(row['photo'])))
        # if os.path.basename(row['photo']) not in self.images:
        #     x = Image.open(os.path.join(dst, os.path.basename(row['photo'])))
        #     self.images[os.path.basename(row['photo'])] = np.array(x)
        # else:
        #     x = Image.fromarray(self.images[os.path.basename(row['photo'])])
        if self.transform is not None:
            x = self.transform(x)
        y = torch.zeros(len(cats))
        y[row['cats']] = 1
        return x, y


if __name__ == '__main__':
    device = 'cuda'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = CSVDataSet(
        'csvs/teamchadchart.csv',
        transforms.Compose([
            transforms.Resize(args.imsize),
            transforms.CenterCrop(args.imsize),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_dataset, val_dataset = random_split(dataset, [.8, .2], generator=torch.Generator().manual_seed(42))
    trainloader = DataLoader(train_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.workers,
                             pin_memory=False,
                             # sampler=
                             )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        num_workers=args.workers,
        pin_memory=False,
        # sampler=
    )

    model = timm.create_model('mobilenetv3_small_075', pretrained=True, num_classes=len(cats))
    model.to(device)
    sgmd = nn.Sigmoid()
    optimizer = torch.optim.SGD(model.parameters(), 0.04,
                                momentum=0.9,
                                weight_decay=1e-4,
                                )
    scheduler = MultiStepLR(optimizer, [10, 20])
    criterion = nn.BCELoss().to(device)

    best_acc = 0
    for epoch in range(args.num_epochs):
        print('Epoch:', epoch + 1)
        model.train()
        progbar = Progbar(len(trainloader))
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(sgmd(outputs), targets)
            try:
                loss.backward()
            except RuntimeError:
                print(' - error', inputs.size(), targets.size())
                continue
            predicted = sgmd(outputs) > 0.5
            optimizer.step()
            optimizer.zero_grad()
            suffix = [('loss', loss.item()),
                      ('acc', predicted.eq(targets).sum().item() / targets.size(0) / len(cats)),
                      ]
            progbar.update(batch_idx + 1, suffix)
        scheduler.step()
        model.eval()
        test_loss = 0
        progbar = Progbar(len(val_loader))
        accuracy = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(sgmd(outputs), targets)
                test_loss += loss.item()
                predicted = sgmd(outputs) > 0.5
                acc = predicted.eq(targets).sum().item() / targets.size(0) / len(cats)
                accuracy += acc
                # f1 = f1_score(gt.float().cpu().numpy(), predicted.float().cpu().numpy(), average='macro')
                suffix = [('loss', loss.item()),
                          ('acc', acc),
                          # ('f1', f1),
                          ]
                progbar.update(batch_idx + 1, suffix)
        acc = accuracy / len(val_loader)
        state = {
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        os.makedirs(f'{args.outdir}/{args.model}', exist_ok=True)
        if acc > best_acc:
            torch.save(state, f'{args.outdir}/{args.model}/best.t7')
            best_acc = acc
        torch.save(model.state_dict(), f'{args.outdir}/{args.model}/temp.t7')

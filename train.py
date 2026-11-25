'''
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from data import load
from tmc import TMC,tmc_loss


@torch.inference_mode()
def score(model,data):
    model.eval()
    loss=correct=0
    for a,b,y in DataLoader(data,batch_size=64):
        p=model(a,b)['log_final']
        correct+=(p.argmax(1)==y).sum().item()
        loss+=F.nll_loss(p,y,reduction='sum').item()
    return loss/len(data),correct/len(data)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['train','eval'])
    parser.add_argument('--dataset',choices=['har','mfeat'],default='har')
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args()
    ckpt=args.checkpoint or Path(__file__).parent/'runs'/args.dataset/'best.pt'
    name=args.dataset
    if args.command=='eval':
        name=torch.load(ckpt,weights_only=True)['dataset']
    torch.manual_seed(42)
    torch.set_num_threads(1)
    (train,val,test),mean,std=load(name,42)
    a,b,y=train.tensors
    dims=(a.size(1),b.size(1),int(y.max())+1)
    model=TMC(*dims)
    if args.command=='train':
        ckpt.parent.mkdir(parents=True,exist_ok=True)
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        best=float('inf')
        loader=DataLoader(train,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(42))
        for epoch in range(1,args.epochs+1):
            model.train()
            for a,b,y in loader:
                opt.zero_grad()
                tmc_loss(model(a,b),y)['total'].backward()
                opt.step()
            nll,acc=score(model,val)
            print(epoch,'val',nll,acc,flush=True)
            if nll<best:
                best=nll
                torch.save(dict(model=model.state_dict(),dims=dims,epoch=epoch,dataset=name,mean=mean,std=std),ckpt)
    model.load_state_dict(torch.load(ckpt,weights_only=True)['model'])
    print('test',score(model,test))


if __name__=='__main__':
    main()
'''
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from data import load
from tmc import TMC, tmc_loss


@torch.inference_mode()
def score(model, data):
    model.eval()
    loss = correct = 0
    for *xs, y in DataLoader(data, batch_size=64):
        p = model(*xs)['log_final']
        correct += (p.argmax(1) == y).sum().item()
        loss += F.nll_loss(p, y, reduction='sum').item()
    return loss / len(data), correct / len(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'eval'])
    parser.add_argument('--dataset', choices=['har', 'mfeat', 'cmu-mosi'], default='har')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    ckpt = args.checkpoint or Path(__file__).parent / 'runs' / args.dataset / 'best.pt'
    name = args.dataset
    if args.command == 'eval':
        name = torch.load(ckpt, weights_only=True)['dataset']
    torch.manual_seed(42)
    torch.set_num_threads(1)
    (train, val, test), mean, std = load(name, 42)
    *xs, y = train.tensors
    input_dims = tuple(x.size(1) for x in xs)
    num_classes = int(y.max()) + 1
    model = TMC(input_dims, num_classes)
    if args.command == 'train':
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        best = float('inf')
        loader = DataLoader(train, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(42))
        for epoch in range(1, args.epochs + 1):
            model.train()
            for *xs, y in loader:
                opt.zero_grad()
                tmc_loss(model(*xs), y)['total'].backward()
                opt.step()
            nll, acc = score(model, val)
            print(epoch, 'val', nll, acc, flush=True)
            if nll < best:
                best = nll
                torch.save(dict(model=model.state_dict(), input_dims=input_dims, num_classes=num_classes,
                                epoch=epoch, dataset=name, mean=mean, std=std), ckpt)
    model.load_state_dict(torch.load(ckpt, weights_only=True)['model'])
    print('test', score(model, test))


if __name__ == '__main__':
    main()
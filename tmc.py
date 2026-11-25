'''
import torch
from torch import nn
from torch.nn import functional as F


class Modality(nn.Module):
    def __init__(self, inputs, dim, classes):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(inputs, dim), nn.GELU(), nn.Linear(dim, 2*dim))
        self.classifier = nn.Linear(dim, classes)
        self.confidence = nn.Linear(dim, 1)
        self.shared = nn.Linear(dim, dim)
        self.unique = nn.Linear(dim, dim)
        self.tokens = nn.Linear(dim, 4*dim)

    def forward(self, x):
        mu, lv = self.encoder(x).chunk(2, -1)
        lv = lv.clamp(-20, 12)
        h = mu + (lv/2).exp()*torch.randn_like(mu) if self.training else mu
        return dict(mu=mu, lv=lv, logits=self.classifier(h), c=self.confidence(h).sigmoid(),
                    r=self.shared(h), u=self.unique(h), t=self.tokens(h).reshape(len(x),4,-1))


def mixture(p,c):
    return torch.logsumexp(p + (c+1e-8).log().log_softmax(1).unsqueeze(-1),1)


class TMC(nn.Module):
    def __init__(self, dim1, dim2, num_classes, dim=64):
        super().__init__()
        self.modalities = nn.ModuleList(Modality(d,dim,num_classes) for d in (dim1,dim2))
        self.cross = nn.MultiheadAttention(dim,4,batch_first=True)
        self.project = nn.Sequential(nn.Linear(2*dim,dim),nn.GELU(),nn.Linear(dim,dim))
        self.classifiers = nn.ModuleList(nn.Linear(dim,num_classes) for _ in range(4))
        self.confidences = nn.ModuleList(nn.Linear(dim,1) for _ in range(4))

    def synergy(self,a,b):
        left = self.cross(a,b,b,need_weights=False)[0].mean(1)
        right = self.cross(b,a,a,need_weights=False)[0].mean(1)
        return self.project(torch.cat((left,right),1))

    def forward(self,x1,x2):
        a,b = self.modalities[0](x1),self.modalities[1](x2)
        ca,cb = a['c']+1e-8,b['c']+1e-8
        r = (ca*a['r']+cb*b['r'])/(ca+cb)
        ta,tb = a['t'],b['t']
        features = (r,a['u'],b['u'],self.synergy(ta,tb))
        logits = [head(x) for head,x in zip(self.classifiers,features)]
        p = torch.stack(logits,1).log_softmax(-1)
        c = torch.cat([head(x).sigmoid() for head,x in zip(self.confidences,features)],1)
        out = dict(log_final=mixture(p,c),log_probs=p,confidence=c)
        if self.training:
            aux = [logits[0]]+[self.classifiers[i](features[i].detach()) for i in (1,2,3)]
            out.update(modalities=(a,b),features=features,aux=torch.stack(aux,1),
                       single=(self.synergy(ta,torch.zeros_like(tb)),self.synergy(torch.zeros_like(ta),tb)))
        return out


def decorrelation(a,b):
    def scale(x):
        return (x-x.mean(0))/(x.std(0)+1e-6)
    return ((scale(a).T@scale(b))/(len(a)-1)).square().mean()


def tmc_loss(out,y):
    if len(y)<2:
        raise ValueError('Training requires at least two samples')
    a,b = out['modalities']
    r,u1,u2,s = out['features']
    p,c = out['log_probs'],out['confidence']
    targets = y[:,None].expand(-1,4)
    ib_terms,errors = [],[]
    for m in (a,b):
        lv = m['lv']
        kl = .5*(m['mu'].square()+lv.expm1()-lv).sum(1)
        beta = .0001+.0009*(1-m['c'].detach().flatten())
        ib_terms.append((kl*beta).mean()+F.cross_entropy(m['logits'],y))
        tcp_target = m['logits'].detach().softmax(-1).gather(1,y[:,None])
        errors.append((m['c']-tcp_target).square())
    ib = sum(ib_terms)
    def nce(student,teacher):
        scores = F.normalize(student)@F.normalize(teacher.detach()).T/.1
        return F.cross_entropy(scores,torch.arange(len(y),device=y.device),reduction='none')
    gap = (a['c']-b['c']).detach().flatten()
    align = (gap.relu()*nce(b['r'],a['r'])+(-gap).relu()*nce(a['r'],b['r'])).mean()
    pairs = ((a['r'],u1),(b['r'],u2),(u1,u2),(s,r),(s,u1),(s,u2))
    dep = sum(v.square().sum(1).mean() for v in out['single'])/2
    dis = sum(decorrelation(x,z) for x,z in pairs)/6 + dep
    ru = p[:,:3].detach()
    weights = c.detach()
    before = mixture(ru,weights[:,:3])
    after = mixture(torch.cat((ru,p[:,3:]),1),weights)
    gain = (before-after).gather(1,y[:,None]).relu().mean()
    cls = F.nll_loss(out['log_final'],y)+.1*F.cross_entropy(out['aux'].transpose(1,2),targets)
    errors.append((c-p.detach().gather(2,targets.unsqueeze(2)).squeeze(2).exp()).square())
    tcp = torch.cat(errors,1).mean()
    return dict(total=cls+ib+.1*align+.01*dis+.1*gain+tcp,cls=cls,ib=ib,align=align,dis=dis,gain=gain,tcp=tcp,dep=dep)
'''
import torch
from torch import nn
from torch.nn import functional as F


class Modality(nn.Module):
    def __init__(self, inputs, dim, classes):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(inputs, dim), nn.GELU(), nn.Linear(dim, 2 * dim))
        self.classifier = nn.Linear(dim, classes)
        self.confidence = nn.Linear(dim, 1)
        self.shared = nn.Linear(dim, dim)
        self.unique = nn.Linear(dim, dim)
        self.tokens = nn.Linear(dim, 4 * dim)

    def forward(self, x):
        mu, lv = self.encoder(x).chunk(2, -1)
        lv = lv.clamp(-20, 12)
        h = mu + (lv / 2).exp() * torch.randn_like(mu) if self.training else mu
        return dict(mu=mu, lv=lv, logits=self.classifier(h), c=self.confidence(h).sigmoid(),
                    r=self.shared(h), u=self.unique(h), t=self.tokens(h).reshape(len(x), 4, -1))


def mixture(p, c):
    return torch.logsumexp(p + (c + 1e-8).log().log_softmax(1).unsqueeze(-1), 1)


class TMC(nn.Module):
    def __init__(self, input_dims, num_classes, dim=64):
        super().__init__()
        if len(input_dims) < 2:
            raise ValueError('TMC requires at least two modalities')
        n_mod = len(input_dims)
        n_heads = n_mod + 2  # [r, u_1, ..., u_M, s]
        self.modalities = nn.ModuleList(Modality(d, dim, num_classes) for d in input_dims)
        self.cross = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.project = nn.Sequential(nn.Linear(n_mod * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.classifiers = nn.ModuleList(nn.Linear(dim, num_classes) for _ in range(n_heads))
        self.confidences = nn.ModuleList(nn.Linear(dim, 1) for _ in range(n_heads))

    def synergy(self, tokens):
        parts = []
        for i, q in enumerate(tokens):
            kv = torch.cat([t for j, t in enumerate(tokens) if j != i], dim=1)
            parts.append(self.cross(q, kv, kv, need_weights=False)[0].mean(1))
        return self.project(torch.cat(parts, dim=1))

    def forward(self, *xs):
        if len(xs) != len(self.modalities):
            raise ValueError(f'Expected {len(self.modalities)} modalities, got {len(xs)}')
        ms = tuple(mod(x) for mod, x in zip(self.modalities, xs))
        cs = [m['c'] + 1e-8 for m in ms]
        r = sum(c * m['r'] for c, m in zip(cs, ms)) / sum(cs)
        tokens = [m['t'] for m in ms]
        features = (r, *(m['u'] for m in ms), self.synergy(tokens))
        logits = [head(x) for head, x in zip(self.classifiers, features)]
        p = torch.stack(logits, 1).log_softmax(-1)
        c = torch.cat([head(x).sigmoid() for head, x in zip(self.confidences, features)], 1)
        out = dict(log_final=mixture(p, c), log_probs=p, confidence=c)
        if self.training:
            aux = [logits[0]] + [self.classifiers[i](features[i].detach()) for i in range(1, len(features))]
            single = tuple(
                self.synergy([t if j == i else torch.zeros_like(t) for j, t in enumerate(tokens)])
                for i in range(len(tokens))
            )
            out.update(modalities=ms, features=features, aux=torch.stack(aux, 1), single=single)
        return out


def decorrelation(a, b):
    def scale(x):
        return (x - x.mean(0)) / (x.std(0) + 1e-6)
    return ((scale(a).T @ scale(b)) / (len(a) - 1)).square().mean()


def tmc_loss(out, y):
    if len(y) < 2:
        raise ValueError('Training requires at least two samples')
    ms = out['modalities']
    n_mod = len(ms)
    r, *us, s = out['features']
    p, c = out['log_probs'], out['confidence']
    n_heads = p.size(1)
    targets = y[:, None].expand(-1, n_heads)

    ib_terms, errors = [], []
    for m in ms:
        lv = m['lv']
        kl = .5 * (m['mu'].square() + lv.expm1() - lv).sum(1)
        beta = .0001 + .0009 * (1 - m['c'].detach().flatten())
        ib_terms.append((kl * beta).mean() + F.cross_entropy(m['logits'], y))
        tcp_target = m['logits'].detach().softmax(-1).gather(1, y[:, None])
        errors.append((m['c'] - tcp_target).square())
    ib = sum(ib_terms)

    def nce(student, teacher):
        scores = F.normalize(student) @ F.normalize(teacher.detach()).T / .1
        return F.cross_entropy(scores, torch.arange(len(y), device=y.device), reduction='none')

    align_terms = []
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            gap = (ms[i]['c'] - ms[j]['c']).detach().flatten()
            align_terms.append(
                (gap.relu() * nce(ms[j]['r'], ms[i]['r'])
                 + (-gap).relu() * nce(ms[i]['r'], ms[j]['r'])).mean()
            )
    align = sum(align_terms) / len(align_terms)

    pairs = [(m['r'], u) for m, u in zip(ms, us)]
    pairs += [(us[i], us[j]) for i in range(n_mod) for j in range(i + 1, n_mod)]
    pairs += [(s, r)] + [(s, u) for u in us]
    dep = sum(v.square().sum(1).mean() for v in out['single']) / len(out['single'])
    dis = sum(decorrelation(x, z) for x, z in pairs) / len(pairs) + dep

    ru = p[:, :-1].detach()
    weights = c.detach()
    before = mixture(ru, weights[:, :-1])
    after = mixture(torch.cat((ru, p[:, -1:]), 1), weights)
    gain = (before - after).gather(1, y[:, None]).relu().mean()

    cls = F.nll_loss(out['log_final'], y) + .1 * F.cross_entropy(out['aux'].transpose(1, 2), targets)
    errors.append((c - p.detach().gather(2, targets.unsqueeze(2)).squeeze(2).exp()).square())
    tcp = torch.cat(errors, 1).mean()
    return dict(total=cls + ib +  align +  dis +  gain + tcp,
                cls=cls, ib=ib, align=align, dis=dis, gain=gain, tcp=tcp, dep=dep)
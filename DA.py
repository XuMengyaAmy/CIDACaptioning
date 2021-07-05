
'''
Paper: Class-Incremental Domain Adaptation with Smoothing and Calibration for Surgical Report Generation
'''
import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
import torch.nn.functional as F
from tqdm import tqdm

import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
import warnings
warnings.filterwarnings("ignore")
import os, json



seed = 1234
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def evaluate_metrics(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)  
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]   
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    if not os.path.exists('predict_caption'):
        os.makedirs('predict_caption')
    json.dump(gen, open('predict_caption/DA.json', 'w'))
    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, loss_fn):
    # Training with cross-entropy
    model.train()
    scheduler.step()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, captions) in enumerate(dataloader):
            detections, captions = detections.to(device), captions.to(device)
            out = model(detections, captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss_labelsmoothing = loss_fn(out, captions_gt) 
            loss_labelsmoothing.backward()
            optim.step()
            this_loss = loss_labelsmoothing.item()
            running_loss += this_loss
            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()

    loss = running_loss / len(dataloader)
    
    return loss



class CELossWithLS(torch.nn.Module):
    def __init__(self, classes=None, smoothing=0.1, gamma=3.0, isCos=True, ignore_index=-1):
        super(CELossWithLS, self).__init__()
        self.complement = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        with torch.no_grad():
            oh_labels = F.one_hot(target.to(torch.int64), num_classes = self.cls).permute(0,1,2).contiguous()
            smoothen_ohlabel = oh_labels * self.complement + self.smoothing / self.cls

        logs = self.log_softmax(logits[target!=self.ignore_index])
        pt = torch.exp(logs)
        return -torch.sum((1-pt).pow(self.gamma)*logs * smoothen_ohlabel[target!=self.ignore_index], dim=1).mean()


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='Incremental domain adaptation for surgical report generation')
    parser.add_argument('--exp_name', type=str, default='m2_transformer')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--m', type=int, default=40)   
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str)

    parser.add_argument('--ls', type=str, default = 'False', help='use label smoothing')

    # # CBS ARGS
    parser.add_argument('--cbs', type=str, default='False')
    parser.add_argument('--kernel_sizex', default=3, type=int)
    parser.add_argument('--kernel_sizey', default=1, type=int)
    parser.add_argument('--decay_epoch', default=2, type=int) 
    parser.add_argument('--std_factor', default=0.9, type=float) 
    args = parser.parse_args()
    print(args)
    print('ls',args.ls)
    print('cbs',args.cbs)
    print('DA Training')

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=6, load_in_tmp=False)  

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, args.features_path, args.annotation_folder, args.annotation_folder)  
    train_dataset, val_dataset = dataset.splits   
    print('train:', len(train_dataset), 'val:', len(val_dataset))

    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=2)  
        pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))

    print(len(text_field.vocab))
    print(text_field.vocab.stoi)


    # Model and dataloaders
    if args.cbs == 'True':
        from models.transformer import MemoryAugmentedEncoder_CBS
        print("MemoryAugmentedEncoder_CBS")
        encoder = MemoryAugmentedEncoder_CBS(3, 0, attention_module=ScaledDotProductAttentionMemory, attention_module_kwargs={'m': args.m})
    else:
        print("MemoryAugmentedEncoder")
        encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory, attention_module_kwargs={'m': args.m}) 
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
 
    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    print('dic_train:',len(dict_dataset_train))   

    ref_caps_train = list(train_dataset.text) 
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    print('dic_val',len(dict_dataset_val))   


    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)


    # Load the pretrained model
    pretrained_params = torch.load('checkpoints/inc_sup_cbs____cbs_ls/%s_best.pth' % args.exp_name)
    model.load_state_dict(pretrained_params['state_dict'], strict = False)
    print("Epoch %d" % pretrained_params['epoch'])  
    print(pretrained_params['best_cider'])

    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)

    if args.ls  == 'True': 
        print('smoothing = 0.1')
        loss_ls_v2 = CELossWithLS(classes=len(text_field.vocab), smoothing=0.1, gamma=0.0, isCos=False, ignore_index=text_field.vocab.stoi['<pad>'])
    else:
        print('smoothing = 0.0')
        loss_ls_v2 = CELossWithLS(classes=len(text_field.vocab), smoothing=0.0, gamma=0.0, isCos=False, ignore_index=text_field.vocab.stoi['<pad>']) 
    
    
    best_cider = .0
    best_bleu = .0
    start_epoch = 0
    best_epoch = 0


    print("Training starts")

    for e in range(start_epoch, start_epoch + 100):  
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        
        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size, shuffle=True,num_workers=args.workers)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size)
    
        ################################   CBS  #################################
        if args.cbs == 'True':
            model.encoder.get_new_kernels(e, args.kernel_sizex, args.kernel_sizey, args.decay_epoch, args.std_factor) 
            model = model.to(device)        
        
        train_loss = train_xe(model, dataloader_train, optim, loss_ls_v2)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field)  
        val_cider = scores['CIDEr']
     

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_bleu = scores['BLEU'][0]
            best_cider = val_cider
            best_epoch = e

            best = True

        print("Validation scores", scores, 'Best epoch',best_epoch,'Best bleu:%.4f, cider:%.4f'%(best_bleu,best_cider))
   
        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_cider': best_cider,
           
        }, 'saved_models/%s_last.pth' % args.exp_name)   

        if best:
            print('saving best epoch...!')
            copyfile('saved_models/%s_last.pth' % args.exp_name, 'saved_models/%s_best.pth' % args.exp_name)


    data = torch.load('saved_models/m2_transformer_best.pth')
    model.load_state_dict(data['state_dict'])
    print("Epoch %d" % data['epoch'])  
    print(data['best_cider'])







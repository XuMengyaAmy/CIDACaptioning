'''
Paper : Class-Incremental Domain Adaptation with Smoothing and Calibration for Surgical Report Generation
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
    with tqdm(desc='evaluation', unit='it', total=len(dataloader)) as pbar:
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
    json.dump(gen, open('predict_caption/predict_caption_val.json', 'w'))

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores



if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='Incremental domain adaptation for surgical report generation')
    parser.add_argument('--exp_name', type=str, default='m2_transformer')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--m', type=int, default=40)   
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--features_path_DA', type=str)
    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--annotation_folder_DA', type=str)

    # CBS ARGS
    parser.add_argument('--cbs', type=str, default=False)
    parser.add_argument('--kernel_sizex', default=3, type=int)
    parser.add_argument('--kernel_sizey', default=1, type=int)
    parser.add_argument('--decay_epoch', default=2, type=int) 
    parser.add_argument('--std_factor', default=0.9, type=float) 

    args = parser.parse_args()
    print(args)
   
    print('Validation')

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=6, load_in_tmp=False)  
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, args.features_path, args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset = dataset.splits   
    print('train:', len(train_dataset))
    print('val:', len(val_dataset))
    

    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=2)  
        pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))

    print('vocabulary size is:', len(text_field.vocab))
    print(text_field.vocab.stoi)

    
    if args.cbs == 'True':
        from models.transformer import MemoryAugmentedEncoder_CBS
        print("MemoryAugmentedEncoder_CBS")
        encoder = MemoryAugmentedEncoder_CBS(3, 0, attention_module=ScaledDotProductAttentionMemory, attention_module_kwargs={'m': args.m})
    else:
        print("MemoryAugmentedEncoder")
        encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory, attention_module_kwargs={'m': args.m}) 

    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    if args.cbs == 'True':
        model.encoder.get_new_kernels(0, args.kernel_sizex, args.kernel_sizey, args.decay_epoch, args.std_factor)
        model = model.to(device)

    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})


    # data = torch.load('checkpoints/inc_sup_cbs____cbs_ls/%s_best.pth' % args.exp_name)
    data = torch.load('checkpoints/few_shot_inc_sup_cbs____cbs_ls/%s_best.pth' % args.exp_name)
   
    model.load_state_dict(data['state_dict'])
    print("Epoch %d" % data['epoch'])  
    print(data['best_cider'])


    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size)
    scores = evaluate_metrics(model, dict_dataloader_val, text_field)  
    print("Validation scores :", scores)







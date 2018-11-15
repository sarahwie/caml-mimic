"""
    Main training code. Loads data, builds the model, trains, tests, evaluates, writes outputs, etc.
"""
import torch
from torch.autograd import Variable, set_grad_enabled

import csv
import argparse
import os 
import numpy as np
import operator
import random
import sys
sys.path.append('..')
import time
from tqdm import tqdm
from collections import defaultdict

from constants import *
print(MODEL_DIR)
print(DATA_DIR)
import datasets
import evaluation
import interpret
import persistence
import learn.models as models
import learn.tools as tools

import git

def main(args):
    assert '-' not in args.criterion
    start = time.time()
    args, model, optimizer, params, dicts, start_epoch = init(args)
    epochs_trained = train_epochs(args, model, optimizer, params, dicts, start_epoch)
    if args.reload_model:
        print("TOTAL ELAPSED TIME FOR %s MODEL AND %d (ADDITIONAL) EPOCHS (hours): %f" % (args.model, epochs_trained, ((time.time() - start)/60/60)))
        print("TOTAL EPOCHS:", args.n_epochs)
    else:
        print("TOTAL ELAPSED TIME FOR %s MODEL AND %d EPOCHS (hours): %f" % (args.model, epochs_trained, ((time.time() - start)/60/60)))

def init(args):
    """
        Load data, build model, create optimizer, create vars to hold metrics, etc.
    """
    #need to handle really large text fields
    csv.field_size_limit(sys.maxsize)

    #load vocab and other lookups
    desc_embed = args.lmbda > 0
    print("loading lookups...")
    dicts = datasets.load_lookups(args, desc_embed=desc_embed)

    META_TEST = args.test_model is not None
    model, start_epoch, optimizer = tools.pick_model(args, dicts, META_TEST)
    print(model)

    params = tools.make_param_dict(args)
    
    return args, model, optimizer, params, dicts, start_epoch

def train_epochs(args, model, optimizer, params, dicts, start_epoch):
    """
        Main loop. does train and test
    """
    metrics_hist = defaultdict(lambda: [])
    metrics_hist_te = defaultdict(lambda: [])
    metrics_hist_tr = defaultdict(lambda: [])
    META_TEST = args.test_model is not None
    test_only = args.test_model is not None
    evaluate = args.test_model is not None
    #train for n_epochs unless criterion metric does not improve for [patience] epochs
    for epoch in range(start_epoch, args.n_epochs): #only train for _x_ more epochs from best-saved model*
        #only test on train/test set on very last epoch
        if epoch == start_epoch and not args.test_model:
            model_dir = os.path.join(MODEL_DIR, '_'.join([args.model, time.strftime('%b_%d_%H:%M', time.localtime())]))
            os.mkdir(model_dir)

            #save model versioning (git) info:
            repo = git.Repo(search_parent_directories=True)
            branch = repo.active_branch.name
            print("branch:", branch)
            sha = repo.head.object.hexsha
            print("SHA hash:", sha) 
            persistence.save_git_versioning_info(model_dir, (branch, sha, args.description))

        elif args.test_model:
            model_dir = os.path.dirname(os.path.abspath(args.test_model))

        metrics_all = one_epoch(args, model, optimizer, epoch, test_only, dicts, model_dir, META_TEST)
        for name in metrics_all[0].keys():
            metrics_hist[name].append(metrics_all[0][name])
        for name in metrics_all[1].keys():
            metrics_hist_te[name].append(metrics_all[1][name])
        for name in metrics_all[2].keys():
            metrics_hist_tr[name].append(metrics_all[2][name])
        metrics_hist_all = (metrics_hist, metrics_hist_te, metrics_hist_tr)

        #save metrics, model, params
        assert '-' not in args.criterion
        persistence.save_everything(args, metrics_hist_all, model, model_dir, params, optimizer, evaluate)

        if test_only:
            #we're done
            break

        if args.criterion in metrics_hist.keys():
            if early_stop(metrics_hist, args.criterion, args.patience):
                #stop training, do tests on test and train sets, and then stop the script
                print("%s hasn't improved in %d epochs, early stopping..." % (args.criterion, args.patience))
                test_only = True
                test_m = [o for o in os.listdir(model_dir) if 'model_best' in o]
                assert(len(test_m) == 1)
                args.test_model = os.path.join(model_dir, test_m[0])
                model = tools.pick_model(args, dicts, META_TEST)
    return epoch-start_epoch+1

def early_stop(metrics_hist, criterion, patience):
    if not np.all(np.isnan(metrics_hist[criterion])):
        if len(metrics_hist[criterion]) >= patience: #only evaluate if have trained for \geq the patience amt of epochs*
            if criterion == 'loss_dev': 
                return np.nanargmin(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
            else:
                return np.nanargmax(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
    else:
        #keep training if criterion results have all been nan so far
        return False
        

def one_epoch(args, model, optimizer, epoch, testing, dicts, model_dir, META_TEST):
    """
        Wrapper to do a training epoch and test on dev
    """

    if not testing:
        set_grad_enabled(True)
        losses, unseen_code_inds = train(args, model, optimizer, epoch, dicts)
        loss = np.mean(losses)
        print("epoch loss: " + str(loss))
    else:
        loss = np.nan
        if model.lmbda > 0:
            #still need to get unseen code inds
            print("getting set of codes not in training set")
            c2ind = dicts['c2ind']
            unseen_code_inds = set(dicts['ind2c'].keys())
            num_labels = len(dicts['ind2c'])
            with open(args.data_path, 'r') as f:
                r = csv.reader(f)
                #header
                next(r)
                for row in r:
                    unseen_code_inds = unseen_code_inds.difference(set([c2ind[c] for c in row[3].split(';') if c != '']))
            print("num codes not in train set: %d" % len(unseen_code_inds))
        else:
            unseen_code_inds = set()

    fold = 'test' if args.version == 'mimic2' else 'dev'
    if epoch == args.n_epochs - 1:
        print("last epoch: testing on test and train sets")
        testing = True

    if not META_TEST:
        #test on dev
        metrics = test(args, model, epoch, fold, unseen_code_inds, dicts, model_dir,
                   testing)
    else:
        metrics = defaultdict(float)

    if testing or epoch == args.n_epochs - 1:
        print("\nevaluating on test")
        metrics_te = test(args, model, epoch, "test", unseen_code_inds, dicts, 
                          model_dir, True)
    else:
        metrics_te = defaultdict(float)
        fpr_te = defaultdict(lambda: [])
        tpr_te = defaultdict(lambda: [])
    metrics_tr = {'loss': loss}
    metrics_all = (metrics, metrics_te, metrics_tr)
    return metrics_all


def train(args, model, optimizer, epoch, dicts):
    """
        Training loop.
        output: losses for each example for this iteration
    """
    print("EPOCH %d" % epoch)
    num_labels = len(dicts['ind2c'])

    losses = []
    #how often to print some info to stdout
    print_every = 25

    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    unseen_code_inds = set(ind2c.keys())
    desc_embed = model.lmbda > 0

    model.train()
    gen = datasets.data_generator(args.data_path, dicts, args.batch_size, num_labels, desc_embed=desc_embed, version=args.version)

    for batch_idx, tup in tqdm(enumerate(gen)):

        old_word_embeds = model.embed.weight.data.cpu().numpy()

        data, target, _, code_set, descs = tup
        data, target = Variable(torch.LongTensor(data)), Variable(torch.FloatTensor(target))
        unseen_code_inds = unseen_code_inds.difference(code_set)
        if args.gpu:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()

        if desc_embed:
            desc_data = descs
        else:
            desc_data = None

        output, loss, _ = model(data, target, desc_data=desc_data)

        loss.backward()
        optimizer.step()

        assert not np.array_equal(model.embed.weight.data.cpu().numpy(), old_word_embeds)
        #if not np.array_equal(model.embed.weight.data.cpu().numpy(), old_word_embeds):
        #	print("Weights updated")
        #else:
        #	print("No update")

        losses.append(loss.item())

        if not args.quiet and batch_idx % print_every == 0:
            #print the average loss of the last 10 batches
            print("Train epoch: {} [batch #{}, batch_size {}, seq length {}]\tLoss: {:.6f}".format(
                epoch, batch_idx, data.size()[0], data.size()[1], np.mean(losses[-10:])))
    return losses, unseen_code_inds

def unseen_code_vecs(model, code_inds, dicts, gpu):
    """
        Use description module for codes not seen in training set.
    """
    code_vecs = tools.build_code_vecs(code_inds, dicts)
    code_inds, vecs = code_vecs
    #wrap it in an array so it's 3d
    desc_embeddings = model.embed_descriptions([vecs], gpu)[0]
    #replace relevant final_layer weights with desc embeddings 
    model.final.weight.data[code_inds, :] = desc_embeddings.data
    model.final.bias.data[code_inds] = 0


def test(args, model, epoch, fold, code_inds, dicts, model_dir, testing):
    """
        Testing loop.
        Returns metrics
    """
    filename = args.data_path.replace('train', fold) 
    print('file for evaluation: %s' % filename)
    num_labels = len(dicts['ind2c'])

    #initialize stuff for saving attention samples
    if args.samples:
        tp_file = open('%s/tp_%s_examples_%d.txt' % (model_dir, fold, epoch), 'w')
        fp_file = open('%s/fp_%s_examples_%d.txt' % (model_dir, fold, epoch), 'w')
        window_size = model.conv.weight.data.size()[2]

    y, yhat, yhat_raw, hids, losses = [], [], [], [], []
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']

    desc_embed = model.lmbda > 0
    if desc_embed and len(code_inds) > 0:
        unseen_code_vecs(model, code_inds, dicts, args.gpu)

    model.eval()
    gen = datasets.data_generator(filename, dicts, 1, num_labels, desc_embed=desc_embed, version=args.version)

    for batch_idx, tup in tqdm(enumerate(gen)):
        data, target, hadm_ids, _, descs = tup
        set_grad_enabled(False)
        data, target = Variable(torch.LongTensor(data)), Variable(torch.FloatTensor(target))
        if args.gpu:
            data = data.cuda()
            target = target.cuda()
        model.zero_grad()

        if desc_embed:
            desc_data = descs
        else:
            desc_data = None

        #get an attention sample for 2% of batches
        get_attn = args.samples and (np.random.rand() < 0.02 or (fold == 'test' and testing))
        output, loss, alpha = model(data, target, desc_data=desc_data, get_attention=get_attn)

        output = output.data.cpu().numpy()
        losses.append(loss.item())
        target_data = target.data.cpu().numpy()
        if get_attn and args.samples:
            interpret.save_samples(data, output, target_data, alpha, window_size, epoch, tp_file, fp_file, dicts=dicts)

        #save predictions, target, hadm ids
        yhat_raw.append(output)
        output = np.round(output)
        y.append(target_data)
        yhat.append(output)
        hids.extend(hadm_ids)

    #close files if needed
    if args.samples:
        tp_file.close()
        fp_file.close()

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)

    #write the predictions
    preds_file = persistence.write_preds(yhat, model_dir, hids, fold, ind2c, yhat_raw)
    #get metrics
    k = 5 if num_labels == 50 else [8,15]
    metrics = evaluation.all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
    evaluation.print_metrics(metrics)
    metrics['loss_%s' % fold] = np.mean(losses)
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a neural network on some clinical documents")
    parser.add_argument("data_path", type=str,
                        help="path to a file containing sorted train data. dev/test splits assumed to have same name format with 'train' replaced by 'dev' and 'test'")
    parser.add_argument("vocab", type=str, help="path to a file holding vocab word list for discretizing words")
    parser.add_argument("Y", type=str, help="size of label space")
    parser.add_argument("model", type=str, choices=["cnn_vanilla", "rnn", "conv_attn", "multi_conv_attn", "saved"], help="model")
    parser.add_argument("n_epochs", type=int, help="number of epochs to train")
    parser.add_argument("--description", type=str, required=False, default="", dest='description', help='provide a description to save with model')
    parser.add_argument("--embed-file", type=str, required=False, dest="embed_file",
                        help="path to a file holding pre-trained embeddings")
    parser.add_argument("--cell-type", type=str, choices=["lstm", "gru"], help="what kind of RNN to use (default: GRU)", dest='cell_type',
                        default='gru')
    parser.add_argument("--rnn-dim", type=int, required=False, dest="rnn_dim", default=128,
                        help="size of rnn hidden layer (default: 128)")
    parser.add_argument("--bidirectional", dest="bidirectional", action="store_const", required=False, const=True,
                        help="optional flag for rnn to use a bidirectional model")
    parser.add_argument("--rnn-layers", type=int, required=False, dest="rnn_layers", default=1,
                        help="number of layers for RNN models (default: 1)")
    parser.add_argument("--embed-size", type=int, required=False, dest="embed_size", default=100,
                        help="size of embedding dimension. (default: 100)")
    parser.add_argument("--filter-size", type=str, required=False, dest="filter_size", default=4,
                        help="size of convolution filter to use. (default: 3) For multi_conv_attn, give comma separated integers, e.g. 3,4,5")
    parser.add_argument("--num-filter-maps", type=int, required=False, dest="num_filter_maps", default=50,
                        help="size of conv output (default: 50)")
    parser.add_argument("--weight-decay", type=float, required=False, dest="weight_decay", default=0,
                        help="coefficient for penalizing l2 norm of model weights (default: 0)")
    parser.add_argument("--lr", type=float, required=False, dest="lr", default=1e-3,
                        help="learning rate for Adam optimizer (default=1e-3)")
    parser.add_argument("--batch-size", type=int, required=False, dest="batch_size", default=16,
                        help="size of training batches")
    parser.add_argument("--dropout", dest="dropout", type=float, required=False, default=0.5,
                        help="optional specification of dropout (default: 0.5)")
    parser.add_argument("--lmbda", type=float, required=False, dest="lmbda", default=0,
                        help="hyperparameter to tradeoff BCE loss and similarity embedding loss. defaults to 0, which won't create/use the description embedding module at all. ")
    parser.add_argument("--dataset", type=str, choices=['mimic2', 'mimic3'], dest="version", default='mimic3', required=False,
                        help="version of MIMIC in use (default: mimic3)")
    parser.add_argument("--test-model", type=str, dest="test_model", required=False, help="path to a saved model to load and evaluate")
    parser.add_argument("--reload-model", type=str, dest="reload_model", required=False, help="path to model to continue training from")
    parser.add_argument("--criterion", type=str, default='f1_micro', required=False, dest="criterion",
                        help="which metric to use for early stopping (default: f1_micro)")
    parser.add_argument("--patience", type=int, default=3, required=False,
                        help="how many epochs to wait for improved criterion metric before early stopping (default: 3)")
    parser.add_argument("--gpu", dest="gpu", action="store_const", required=False, const=True,
                        help="optional flag to use GPU if available")
    parser.add_argument("--public-model", dest="public_model", action="store_const", required=False, const=True,
                        help="optional flag for testing pre-trained models from the public github")
    parser.add_argument("--stack-filters", dest="stack_filters", action="store_const", required=False, const=True,
                        help="optional flag for multi_conv_attn to instead use concatenated filter outputs, rather than pooling over them")
    parser.add_argument("--samples", dest="samples", action="store_const", required=False, const=True,
                        help="optional flag to save samples of good / bad predictions")
    parser.add_argument("--quiet", dest="quiet", action="store_const", required=False, const=True,
                        help="optional flag not to print so much during training")
    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    main(args)

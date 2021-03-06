"""
    Saving relevant things.
"""
import csv
import json
import os

import numpy as np
import torch

from constants import *
from learn import models

def save_metrics(metrics_hist_all, model_dir):
    with open(model_dir + "/metrics.json", 'w') as metrics_file:
        #concatenate dev, train metrics into one dict
        data = metrics_hist_all[0].copy()
        data.update({"%s_te" % (name):val for (name,val) in metrics_hist_all[1].items()})
        data.update({"%s_tr" % (name):val for (name,val) in metrics_hist_all[2].items()})
        json.dump(data, metrics_file, indent=1)

def save_params_dict(params):
    with open(params["model_dir"] + "/params.json", 'w') as params_file:
        json.dump(params, params_file, indent=1)

def save_git_versioning_info(model_dir, info):
    with open(model_dir + "/git_info.txt", 'w') as f:
        f.write("Branch: " + info[0] + '\n')
        f.write("SHA: " + info[1] + '\n')
        f.write("Provided description: " + info[2] + '\n')
           

def write_preds(yhat, model_dir, hids, fold, ind2c, yhat_raw=None):
    """
        INPUTS:
            yhat: binary predictions matrix 
            model_dir: which directory to save in
            hids: list of hadm_id's to save along with predictions
            fold: train, dev, or test
            ind2c: code lookup
            yhat_raw: predicted scores matrix (floats)
    """
    preds_file = "%s/preds_%s.psv" % (model_dir, fold)
    with open(preds_file, 'w') as f:
        w = csv.writer(f, delimiter='|')
        for yhat_, hid in zip(yhat, hids):
            codes = [ind2c[ind] for ind in np.nonzero(yhat_)[0]]
            if len(codes) == 0:
                w.writerow([hid, ''])
            else:
                w.writerow([hid] + list(codes))
    if fold != 'train' and yhat_raw is not None:
        #write top 100 scores so we can re-do @k metrics later
        #top 100 only - saving the full set of scores is very large (~1G for mimic-3 full test set)
        scores_file = '%s/pred_100_scores_%s.json' % (model_dir, fold)
        scores = {}
        sortd = np.argsort(yhat_raw)[:,::-1]
        for i,(top_idxs, hid) in enumerate(zip(sortd, hids)):
            scores[int(hid)] = {ind2c[idx]: float(yhat_raw[i][idx]) for idx in top_idxs[:100]}
        with open(scores_file, 'w') as f:
            json.dump(scores, f, indent=1)
    return preds_file

def save_everything(args, metrics_hist_all, model, model_dir, params, optimizer, evaluate=False):
    """
        Save metrics, model, params all in model_dir
    """
    if not evaluate:
        save_metrics(metrics_hist_all, model_dir)
        params['model_dir'] = model_dir
        save_params_dict(params)

        #save the model with the best criterion metric
        if not np.all(np.isnan(metrics_hist_all[0][args.criterion])):
            
            if args.criterion == 'loss_dev': 
                eval_val = np.nanargmin(metrics_hist_all[0][args.criterion])
            else:
                eval_val = np.nanargmax(metrics_hist_all[0][args.criterion])

            if eval_val == len(metrics_hist_all[0][args.criterion]) - 1:
                sd = model.cpu().state_dict()

                filename = [o for o in os.listdir(model_dir) if 'model_best' in o]
                if len(filename) != 0:  #delete old models
                    assert len(filename) == 1
                    os.remove(os.path.join(model_dir, filename[0])) 

                filename = [o for o in os.listdir(model_dir) if 'optim_best' in o]
                if len(filename) != 0:  #delete old optimizers
                    assert len(filename) == 1
                    os.remove(os.path.join(model_dir, filename[0])) 

                #save new model
                torch.save(sd, model_dir + "/model_best_%s_epoch_%d.pth" % (args.criterion, len(metrics_hist_all[0][args.criterion]) - 1))
                #save optimizer
                torch.save(optimizer.state_dict(), model_dir + "/optim_best_%s_epoch_%d.pth" % (args.criterion, len(metrics_hist_all[0][args.criterion]) - 1))

                if args.gpu:
                    model.cuda()
        print("saved metrics, params, model to directory %s\n" % (model_dir))

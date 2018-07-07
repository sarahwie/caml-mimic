"""
    Holds PyTorch models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform
from torch.autograd import Variable

import numpy as np

from math import floor
import random
import sys
import time

from constants import *
from dataproc import extract_wvs

#TODO: should upgrade to Pytorch 0.4.0 and use torch.where
def where(cond, x_1, x_2):
    cond = cond.float()    
    return (cond * x_1) + ((1-cond) * x_2)

class BaseModel(nn.Module):

    def __init__(self, Y, embed_file, dicts, lmbda=0, dropout=0.5, gpu=True, embed_size=100):
        super(BaseModel, self).__init__()
        torch.manual_seed(1337)
        self.gpu = gpu
        self.Y = Y
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=dropout)
        self.lmbda = lmbda

        #make embedding layer
        if embed_file:
            print("loading pretrained embeddings...")
            W = torch.Tensor(extract_wvs.load_embeddings(embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embed.weight.data = W.clone()
        else:
            #add 2 to include UNK and PAD
            vocab_size = len(dicts['ind2w'])
            self.embed = nn.Embedding(vocab_size+2, embed_size, padding_idx=0)


    def _get_loss(self, yhat, target, diffs=None):
        #calculate the BCE
        loss = F.binary_cross_entropy(yhat, target)

        #add description regularization loss if relevant
        if self.lmbda > 0 and diffs is not None:
            diff = torch.stack(diffs).mean()
            loss = loss + diff
        return loss

    def embed_descriptions(self, desc_data, gpu):
        #label description embedding via convolutional layer
        #number of labels is inconsistent across instances, so have to iterate over the batch
        b_batch = []
        for inst in desc_data:
            if len(inst) > 0:
                if gpu:
                    lt = Variable(torch.cuda.LongTensor(inst))
                else:
                    lt = Variable(torch.LongTensor(inst))
                d = self.desc_embedding(lt)
                d = d.transpose(1,2)
                d = self.label_conv(d)
                d = F.max_pool1d(F.tanh(d), kernel_size=d.size()[2])
                d = d.squeeze(2)
                b_inst = self.label_fc1(d)
                b_batch.append(b_inst)
            else:
                b_batch.append([])
        return b_batch

    def _compare_label_embeddings(self, target, b_batch, desc_data):
        #description regularization loss 
        #b is the embedding from description conv
        #iterate over batch because each instance has different # labels
        diffs = []
        for i,bi in enumerate(b_batch):
            ti = target[i]
            inds = torch.nonzero(ti.data).squeeze().cpu().numpy()

            zi = self.final.weight[inds,:]
            diff = (zi - bi).mul(zi - bi).mean()

            #multiply by number of labels to make sure overall mean is balanced with regard to number of labels
            diffs.append(self.lmbda*diff*bi.size()[0])
        return diffs


class ConvAttnPool(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, lmbda, gpu, dicts, embed_size=100, dropout=0.5):
        super(ConvAttnPool, self).__init__(Y, embed_file, dicts, lmbda, dropout=dropout, gpu=gpu, embed_size=embed_size)

        #initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=floor(kernel_size/2))
        xavier_uniform(self.conv.weight)

        #context vectors for computing attention as in 2.2
                #TODO: JAMES HAD BIAS=TRUE HERE-- make a difference?

        self.U = nn.Linear(num_filter_maps, Y, bias=True)
        xavier_uniform(self.U.weight)

        #final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.final.weight)
        
        #conv for label descriptions as in 2.5
        #description module has its own embedding and convolution layers
        if lmbda > 0:
            W = self.embed.weight.data
            self.desc_embedding = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0) #TODO: padding_idx=0?
            self.desc_embedding.weight.data = W.clone()

            self.label_conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=floor(kernel_size/2))
            xavier_uniform(self.label_conv.weight)

            self.label_fc1 = nn.Linear(num_filter_maps, num_filter_maps)
            xavier_uniform(self.label_fc1.weight)
        
    def forward(self, x, target, desc_data=None, get_attention=True):
        #get embeddings and apply dropout
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        #apply convolution and nonlinearity (tanh)
        x = F.tanh(self.conv(x).transpose(1,2))
        #apply attention
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1,2)), dim=2)
        #document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)
        #final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        
        if desc_data is not None:
            #run descriptions through description module
            b_batch = self.embed_descriptions(desc_data, self.gpu)
            #get l2 similarity loss
            diffs = self._compare_label_embeddings(target, b_batch, desc_data)
        else:
            diffs = None
            
        #final sigmoid to get predictions
        yhat = F.sigmoid(y)
        loss = self._get_loss(yhat, target, diffs)
        return yhat, loss, alpha

class ConvAttnPoolPlusGram(BaseModel):
    def __init__(self, Y, embed_file, code_embed_file, kernel_size, num_filter_maps, lmbda, gpu, dicts, recombine_method, hidden_sim_size=20, embed_size=100, dropout=0.5):
        super(ConvAttnPoolPlusGram, self).__init__(Y, embed_file, dicts, lmbda, dropout=dropout, gpu=gpu, embed_size=embed_size)

        self.embed_size = embed_size
        #make embedding layer
        if code_embed_file:
            print("loading pretrained CODE embeddings...")
            #TODO: UPDATE HERE TO LOAD IN PRETRAINED CODE_EMBEDDINGS FILE
            #TODO: make sure that p.t embeds have dim. n+2*
            #W = torch.Tensor(extract_wvs.load_embeddings(embed_file))
            # self.concept_size = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            # self.embed.weight.data = W.clone()
            raise Exception("*TODO not completed*")

        else:
            #TODO: make sure this is what want**
            print("Catch: NOT using pretrained code embeddings!")
            #TODO: FIX OR REMOVE THIS***
            concepts_size = len(dicts['ind2concept'])+2
            self.concept_embed = nn.Embedding(concepts_size, embed_size, padding_idx=0) #TODO: codes and word embeds must have same dimensionality- insert check for this when loaded from P.T. file**
            #TODO: DO ANYTHING TO INIT. THESE WEIGHTS TO START?*

        #TODO: Here, compute attentional similarity between GRAM embedding and word vector as a measure of 'confidence' in extracted concept?
        #TODO: HYPERPARAM TUNING: HOW MANY LAYERS/DIMS IN THIS F.F. NET FOR CALC. SIM. SCORE??**

        #initialize the feedforward network for computing GRAM embedding sim. score, as well as the attentional distribution over the embeds*
        self.fc1 = nn.Linear(2*embed_size, hidden_sim_size) #= Ed's W_a notation
        self.relu = nn.Tanh() #TODO: TRY DIFF NONLINS. HERE- ED USED TANH**
        self.fc2 = nn.Linear(hidden_sim_size, 1) #= Ed's u_a notation
        #TODO: as per James, init these with xavier_uniform (as per Glorot & Bengio 2010)
        xavier_uniform(self.fc1.weight)
        xavier_uniform(self.fc2.weight)

        if recombine_method == 'linear_layer':
            self.recombine = nn.Linear(2*embed_size, embed_size, bias=True)
            xavier_uniform(self.recombine.weight)

        elif recombine_method == 'weight_matrix':
            self.SM = nn.Softmax(dim=2)
            self.recombine = nn.Embedding(len(dicts['concept_word'])+2, 2, padding_idx=0)
            #xavier_uniform(self.recombine.weight) #TODO: not initializing with this b/c destroys the 0's embedding in pad position/not necessary for embeds (?)

        elif recombine_method == 'full_replace':
            pass

        else:
            raise Exception("Argument Error! Recombination of concept and word embeddings")

        #rest of network the same

        # initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=floor(kernel_size / 2))
        xavier_uniform(self.conv.weight)

        # context vectors for computing attention as in 2.2
            #TODO: JAMES HAD BIAS=TRUE HERE-- make a difference?

        self.U = nn.Linear(num_filter_maps, Y, bias=True)
        xavier_uniform(self.U.weight)

        # final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.final.weight)

        # conv for label descriptions as in 2.5
        # description module has its own embedding and convolution layers
        if lmbda > 0:
            W = self.embed.weight.data
            self.desc_embedding = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0) #TODO: padding_idx=0?
            self.desc_embedding.weight.data = W.clone()

            self.label_conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size,
                                        padding=floor(kernel_size / 2))
            xavier_uniform(self.label_conv.weight)

            self.label_fc1 = nn.Linear(num_filter_maps, num_filter_maps)
            xavier_uniform(self.label_fc1.weight)

    def forward(self, data, target, recombine_method, desc_data=None, get_attention=True):

        x, concepts, parents, batched_concepts_mask, dm, word_concept_mask, gpu = data #unpack input

        # get embeddings
        x = self.embed(x)
        x = x.transpose(1, 2)

        #if this isn't a test case with no extracted concepts, proceed
        if batched_concepts_mask is not None:

            # pull parent codes from expanded codeset & embed as 3D matrix
            
            p = self.concept_embed(parents.view(-1, parents.size(2)))

            #reshape
            p = p.view(parents.size(0), parents.size(1), parents.size(2), -1)
            p = p.transpose(1, 3)

            children = p[:, :, 0:1, :].expand(-1,-1,6,-1) #these are the children embeddings
            
            #TODO: CONCAT IN CHILD, ANCESTOR ORDER-- done**
            inpt = torch.cat((children, p),1)
            inpt = inpt.transpose(1,3)

#------------------------------------------------------------------ LEARN CONCEPT EMBEDDINGS

            #reshape input matrix for each codepair
            out = self.fc2(self.relu(self.fc1(inpt))) #out becomes our similarity score, softmax across the 5 dimensions
            # print("Attention Comp. Output:", out.size())

            #now use attn. to construct inpt embeddings--
            alpha = F.softmax(out, dim=2) #across all 6 scores for a set and its parents**

            #Now recombine to produce embedding matrices:
            c = alpha.transpose(2,3).matmul(p.transpose(1,3)).squeeze(2).transpose(1,2)

            # print(c.size()) #matches old concept embedding shape (except only over concepts and not whole input sentence length)

            dm = self.embed(dm)
            dm = dm.transpose(1, 2)

            #compute linear interpolation
            if recombine_method == 'linear_layer':
                concat_mat = torch.cat((c,dm),1) #concat concept and word embeds @ concept posn.'s, input to linear layer:
                linear_interp = self.recombine(concat_mat.transpose(1,2))
                linear_interp = linear_interp.transpose(1,2)

            elif recombine_method == 'weight_matrix':
                concat_mat = torch.cat((c,dm),1)

                # print("Concat Mat. shape:", concat_mat.shape)
                # print("WCM:", word_concept_mask.shape)

                #we softmax this so that it can be a linear layer
                lines = self.SM(self.recombine(word_concept_mask))
                # print("Lines:", lines.shape)
                weighting = torch.cat((lines[:,:,0:1].expand(-1,-1,self.embed_size),lines[:,:,1:2].expand(-1,-1,self.embed_size)),2).transpose(1,2)
                # print("Weighting:", weighting.shape)
                assert weighting.shape == concat_mat.shape

                #do the element-wise multiplication based on scalar factor
                linear_interp = weighting * concat_mat

                #recombine
                linear_interp = linear_interp[:,0:self.embed_size,] + linear_interp[:,self.embed_size:self.embed_size*2,:]

                #TODO: there must be an easier way to do this whole operation....

            elif recombine_method == 'full_replace':
                linear_interp = c
                print("C SHAPE:", c.shape)

#------------------------------------------------------------------- JOIN CONCEPT & WORD MATRICES 

            #TODO: here is where we can add a learnable weighting function for the two embeddings

            concept_mask = batched_concepts_mask.expand(linear_interp.size(1),-1, -1).transpose(0,1) 
            #only pull those concept embeddings which represent the actual positions of the concepts
            concept_embeds = linear_interp[concept_mask] #**row-stacked**

            #get mask over text
            if gpu:
                mask = where(concepts > Variable(torch.zeros(concepts.size())).type(torch.LongTensor).cuda(), Variable(torch.ones(concepts.size())).cuda(), Variable(torch.zeros(concepts.size())).cuda()).type(torch.ByteTensor).cuda()

            else:
                mask = where(concepts > Variable(torch.zeros(concepts.size())).type(torch.LongTensor), Variable(torch.ones(concepts.size())), Variable(torch.zeros(concepts.size()))).type(torch.ByteTensor)

            #reshape/expand along embedding dimension
            mask = mask.expand(x.size(1),-1, -1).transpose(0,1) #represents positions of concept embeddings in text

            #should have same shape as concept_embeds
            assert x[mask].size() == linear_interp[concept_mask].size()

            #do the sub! woohoo: it works :)
            x[mask] = linear_interp[concept_mask]

        #TODO: CONSIDER OVERLAPPING CONCEPTS- HERE HAVE MODIFIED TO ONLY HAVE ONE CONCEPT PER WORD-EMBEDDING-- this is not realistic**
        #TODO: consider to perform dropout earlier on only word vectors, or remove altogether instead of having here.
        x = self.embed_drop(x)

#-------------------------------------------------------------------- STANDARD CAML
        # apply convolution and nonlinearity (tanh)
        x = F.tanh(self.conv(x).transpose(1, 2))
        # apply attention
        #print("U's SHAPE:", self.U.weight.shape)
        #print("SOFTMAX SHAPE:", self.U.weight.matmul(z.transpose(1, 2)).shape)
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        #print(alpha.size())
        # document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)
        #print(m.size())
        # final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        if desc_data is not None:
            # run descriptions through description module
            b_batch = self.embed_descriptions(desc_data, self.gpu)
            # get l2 similarity loss
            diffs = self._compare_label_embeddings(target, b_batch, desc_data)
        else:
            diffs = None

        # final sigmoid to get predictions
        yhat = F.sigmoid(y)
        loss = self._get_loss(yhat, target, diffs)
        return yhat, loss, alpha

class ConvAttnPoolPlusConceptEmbeds(BaseModel):
    def __init__(self, Y, embed_file, code_embed_file, kernel_size, num_filter_maps, lmbda, gpu, dicts, hidden_sim_size=20, embed_size=100, dropout=0.5):
        super(ConvAttnPoolPlusConceptEmbeds, self).__init__(Y, embed_file, dicts, lmbda, dropout=dropout, gpu=gpu, embed_size=embed_size)

        #make embedding layer
        if code_embed_file:
            print("loading pretrained CODE embeddings...")
            #TODO: UPDATE HERE TO LOAD IN PRETRAINED CODE_EMBEDDINGS FILE
            #W = torch.Tensor(extract_wvs.load_embeddings(embed_file))
            # self.concept_size = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            # self.embed.weight.data = W.clone()
            raise Exception("*TODO not completed*")

        else:
            #TODO: make sure this is what want**
            print("Catch: NOT using pretrained code embeddings!")
            concepts_size = len(dicts['ind2concept'])+2
            print("CONCEPTS SIZE:", concepts_size)
            self.concept_embed = nn.Embedding(concepts_size, embed_size, padding_idx=0) #TODO: codes and word embeds must have same dimensionality- insert check for this when loaded from P.T. file**
            #TODO: DO ANYTHING TO INIT. THESE WEIGHTS TO START?*

        #same network as James here

        # initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=floor(kernel_size / 2))
        xavier_uniform(self.conv.weight)

        # context vectors for computing attention as in 2.2
                #TODO: JAMES HAD BIAS=TRUE HERE-- make a difference?

        self.U = nn.Linear(num_filter_maps, Y, bias=True)
        xavier_uniform(self.U.weight)

        # final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.final.weight)

        # conv for label descriptions as in 2.5
        # description module has its own embedding and convolution layers
        if lmbda > 0:
            W = self.embed.weight.data
            self.desc_embedding = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0) #TODO: padding_idx=0?
            self.desc_embedding.weight.data = W.clone()

            self.label_conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size,
                                        padding=floor(kernel_size / 2))
            xavier_uniform(self.label_conv.weight)

            self.label_fc1 = nn.Linear(num_filter_maps, num_filter_maps)
            xavier_uniform(self.label_fc1.weight)

    def forward(self, data, target, desc_data=None, get_attention=True):

        x, concepts = data #unpack input

        # get embeddings
        x = self.embed(x)
        x = x.transpose(1, 2)

        c = self.concept_embed(concepts)
        c = c.transpose(1, 2)

        #RECONSTRUCT THE INPUT EMBEDDING MATRIX BASED ON USING THE CONCEPTS OR WORDS
        #TODO: rewrite this to be performed in matrix form? (no obvious way, can look into l8r)
        #TODO: here is where we can add a learnable weighting function for the two embeddings
        for batch_el in range(c.shape[0]):                
            for word in range(c.shape[2]):
                if word == 0: #first element and first array: create new one
                    if concepts[batch_el, word].data[0] != 0: 
                        patient_embed = c[batch_el, :, word]
                    else:
                        patient_embed = x[batch_el, :, word]
                    patient_embed = patient_embed.view(1,-1,1)

                else: #not first word
                    if concepts[batch_el, word].data[0] != 0: 
                        patient_embed = torch.cat((patient_embed, c[batch_el, :, word].view(1,-1,1)), 2)
                    else:
                        patient_embed = torch.cat((patient_embed, x[batch_el, :, word].view(1,-1,1)), 2)

            #make large patient matrix
            if batch_el == 0:
                z = patient_embed
            else:
                z = torch.cat((z, patient_embed), 0)

        print(z.size())

        #TODO: CONSIDER OVERLAPPING CONCEPTS- HERE HAVE MODIFIED TO ONLY HAVE ONE CONCEPT PER WORD-EMBEDDING-- this is not realistic**
        #TODO: consider to perform dropout earlier on only word vectors, or remove altogether instead of having here.
        z = self.embed_drop(z)

        #THIS PART IS IDENTICAL TO JAMES' MODEL--
        #-------------------------------------------------------------------------------------------------------------
        # apply convolution and nonlinearity (tanh)
        z = F.tanh(self.conv(z).transpose(1, 2))
        print(z.size())
        # apply attention
        alpha = F.softmax(self.U.weight.matmul(z.transpose(1, 2)), dim=2)
        # document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(z)
        # final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        if desc_data is not None:
            # run descriptions through description module
            b_batch = self.embed_descriptions(desc_data, self.gpu)
            # get l2 similarity loss
            diffs = self._compare_label_embeddings(target, b_batch, desc_data)
        else:
            diffs = None

        # final sigmoid to get predictions
        yhat = F.sigmoid(y)
        loss = self._get_loss(yhat, target, diffs)
        return yhat, loss, alpha
    
class VanillaConv(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, gpu=True, dicts=None, embed_size=100, dropout=0.5):
        super(VanillaConv, self).__init__(Y, embed_file, dicts, dropout=dropout, embed_size=embed_size) 
        #initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size)
        xavier_uniform(self.conv.weight)

        #linear output
        self.fc = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.fc.weight)

    def forward(self, x, target, desc_data=None, get_attention=False):
        #embed
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        #conv/max-pooling
        c = self.conv(x)
        if get_attention:
            #get argmax vector too
            x, argmax = F.max_pool1d(F.tanh(c), kernel_size=c.size()[2], return_indices=True)
            attn = self.construct_attention(argmax, c.size()[2])
        else:
            x = F.max_pool1d(F.tanh(c), kernel_size=c.size()[2])
            attn = None
        x = x.squeeze(dim=2)

        #linear output
        x = self.fc(x)

        #final sigmoid to get predictions
        yhat = F.sigmoid(x)
        loss = self._get_loss(yhat, target)
        return yhat, loss, attn

    def construct_attention(self, argmax, num_windows):
        attn_batches = []
        for argmax_i in argmax:
            attns = []
            for i in range(num_windows):
                #generate mask to select indices of conv features where max was i
                mask = (argmax_i == i).repeat(1,self.Y).t()
                #apply mask to every label's weight vector and take the sum to get the 'attention' score
                weights = self.fc.weight[mask].view(-1,self.Y)
                if len(weights.size()) > 0:
                    window_attns = weights.sum(dim=0)
                    attns.append(window_attns)
                else:
                    #this window was never a max
                    attns.append(Variable(torch.zeros(self.Y)).cuda())
            #combine
            attn = torch.stack(attns)
            attn_batches.append(attn)
        attn_full = torch.stack(attn_batches)
        #put it in the right form for passing to interpret
        attn_full = attn_full.transpose(1,2)
        return attn_full


class VanillaRNN(BaseModel):
    """
        General RNN - can be LSTM or GRU, uni/bi-directional
    """

    def __init__(self, Y, embed_file, dicts, rnn_dim, cell_type, num_layers, gpu, embed_size=100, bidirectional=False):
        super(VanillaRNN, self).__init__(Y, embed_file, dicts, embed_size=embed_size, gpu=gpu)
        self.gpu = gpu
        self.rnn_dim = rnn_dim
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        #recurrent unit
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(self.embed_size, floor(self.rnn_dim/self.num_directions), self.num_layers, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(self.embed_size, floor(self.rnn_dim/self.num_directions), self.num_layers, bidirectional=bidirectional)
        #linear output
        self.final = nn.Linear(self.rnn_dim, Y)

        #arbitrary initialization
        self.batch_size = 16
        self.hidden = self.init_hidden()

    def forward(self, x, target, desc_data=None, get_attention=False):
        #clear hidden state, reset batch size at the start of each batch
        self.refresh(x.size()[0])

        #embed
        embeds = self.embed(x).transpose(0,1)
        #apply RNN
        out, self.hidden = self.rnn(embeds, self.hidden)

        #get final hidden state in the appropriate way
        last_hidden = self.hidden[0] if self.cell_type == 'lstm' else self.hidden
        last_hidden = last_hidden[-1] if self.num_directions == 1 else last_hidden[-2:].transpose(0,1).contiguous().view(self.batch_size, -1)
        #apply linear layer and sigmoid to get predictions
        yhat = F.sigmoid(self.final(last_hidden))
        loss = self._get_loss(yhat, target)
        return yhat, loss, None

    def init_hidden(self):
        if self.gpu:
            h_0 = Variable(torch.cuda.FloatTensor(self.num_directions*self.num_layers, self.batch_size,
                                                  floor(self.rnn_dim/self.num_directions)).zero_())
            if self.cell_type == 'lstm':
                c_0 = Variable(torch.cuda.FloatTensor(self.num_directions*self.num_layers, self.batch_size,
                                                      floor(self.rnn_dim/self.num_directions)).zero_())
                return (h_0, c_0)
            else:
                return h_0
        else:
            h_0 = Variable(torch.zeros(self.num_directions*self.num_layers, self.batch_size, floor(self.rnn_dim/self.num_directions)))
            if self.cell_type == 'lstm':
                c_0 = Variable(torch.zeros(self.num_directions*self.num_layers, self.batch_size, floor(self.rnn_dim/self.num_directions)))
                return (h_0, c_0)
            else:
                return h_0

    def refresh(self, batch_size):
        self.batch_size = batch_size
        self.hidden = self.init_hidden()

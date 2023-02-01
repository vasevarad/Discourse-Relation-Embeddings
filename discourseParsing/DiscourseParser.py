import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import csv
import pickle
import sys
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import argparse
import pandas as pd




class DiscourseParser(nn.Module):


    def __init__(self, opt):
        """Initialize the classifier: defines architecture and basic hyper-parameters. """
        super(DiscourseParser, self).__init__()

        ##LSTM parameters:
        self.input_dim = opt.input_dim ##size of sc input embedding 

        self.hidden_dim = opt.hidden_dim   ##size of hidden state vector

        self.num_layer = opt.num_layer ##layers for Suicide LSTM
        self.num_direction = int(opt.num_direction) ##1 or 2 depending if bidirectiona;

        self.dropout_rate=float(opt.dropout) ##rate of dropout, applied between the two LSTMs
        self.is_cuda=opt.cuda  #whether cuda (GPUs) are available
        self.attn_act=opt.attn_act
        self.cell_type=opt.cell_type

        self.word_embedding_dict = pickle.load(open(opt.word_embedding_dict,'rb'))  ##maps words to pre-trained embeddings


        ## Architecture: Word RNN
        if opt.cell_type=='LSTM':
            self.word_RNN = nn.LSTM(self.input_dim, self.hidden_dim, bidirectional=True)
        elif opt.cell_type=='GRU':
            self.word_RNN = nn.GRU(self.input_dim, self.hidden_dim, bidirectional=True)
        else:
            print("Invalid Cell Type:",opt.cell_type)
            sys.exit()
        ## Architecture: Discourse Argument RNN
        if opt.cell_type=='LSTM':
            self.da_RNN = nn.LSTM(self.hidden_dim*2, self.hidden_dim, bidirectional=self.num_direction==2)
        elif opt.cell_type=='GRU':
            self.da_RNN = nn.GRU(self.hidden_dim*2, self.hidden_dim, bidirectional=self.num_direction==2)
        else:
            print("Invalid Cell Type:",self.cell_type)
            sys.exit()

        self.word_attn=nn.Linear(self.num_direction*self.hidden_dim, self.num_direction*self.hidden_dim)
        if opt.attn_type=='element-wise':
            self.word_attn_combine=nn.Linear(self.num_direction*self.hidden_dim, self.num_direction*self.hidden_dim, bias=False)
        elif opt.attn_type=='vector-wise':
            self.word_attn_combine=nn.Linear(self.num_direction*self.hidden_dim, 1, bias=False)
        else:
            print("Invalid attention type",self.attn_type)
            sys.exit()

        self.word_dropout=nn.Dropout(p=self.dropout_rate)

        pdtb_class_num=4
        pdtb_type_num=16
        pdtb_subtype_num=25
        self.das_to_class=nn.Linear(2*self.num_direction*self.hidden_dim,pdtb_class_num) # H_Arg1 (+) H_Arg2 to Four types of discourse relation classes
        self.das_to_type=nn.Linear(2*self.num_direction*self.hidden_dim,pdtb_type_num) # H_Arg1 (+) H_Arg2 to 16 types of discourse relation types
        self.da_to_subtype=nn.Linear(self.num_direction*self.hidden_dim,pdtb_subtype_num) # H_Arg1 (+) H_Arg2 to 25 types of discourse relation subtypes


        ## Check the parameters for the current model
        print("[Model Initialization]:")
        print("Cell Type: "+str(self.cell_type))
        print("Input Dimenstion:",self.input_dim)
        print("Hidden Dimension: " +str(self.hidden_dim))
        print("Hidden Layers: "+str(self.num_layer))
        print("# of Directions for %s :"%(self.cell_type)+str(self.num_direction))
        print("Dropout Rate: "+str(self.dropout_rate))
        print("CUDA Usage: "+str(self.is_cuda))


        if self.is_cuda:
            self.word_RNN = self.word_RNN.cuda()
            self.das_to_class = self.das_to_class.cuda()
            self.das_to_type = self.das_to_type.cuda()
            self.da_to_subtype = self.da_to_subtype.cuda()

    def attn_mul(self,rnn_outputs, attn_weights):
        attn_vectors = None
        for i in range(rnn_outputs.size(0)):
            h_i = rnn_outputs[i]
            a_i = attn_weights[i]
            h_i = a_i * h_i
            h_i = h_i.unsqueeze(0)
            if(attn_vectors is None):
                attn_vectors=h_i
            else:
                attn_vectors = torch.cat((attn_vectors,h_i),0)
        return torch.sum(attn_vectors, 0).unsqueeze(0)

    

    def forward(self, tr_meta, da_embedding_seq):
        """Defines the forward pass through the full deep learning model"""

        case, disCon, arg1Idx, arg2Idx = tr_meta
        # use batch input and get all the hidden vectors and concatenate them

        da_input = []
        empty_seq_da_idxs=[]

        
        for i in range(len(da_embedding_seq)):
            # get da embeddings from the given tweet
            # print(word_embedding_seq)
            word_embedding_seq = []  # Word embeddings in the give discourse argument will be loaded to this list

            for word in da_embedding_seq[i]:  # load the words from 'i'th Discourse Argument
                try:
                    embedding = autograd.Variable(torch.FloatTensor(self.word_embedding_dict[u'' + word.lower()]))
                    word_embedding_seq.append(embedding)
                except:  # ignore UNKs
                    continue
            if len(word_embedding_seq) == 0:
                empty_seq_da_idxs.append(i)  # keep track of indexes of Discourse Arguments with all words missing from pretrained vectors
                continue

            word_embedding_seq = torch.cat(word_embedding_seq).view(len(word_embedding_seq), 1, -1)

            if self.is_cuda:
                word_embedding_seq=word_embedding_seq.cuda()
            if self.cell_type=='LSTM':
                word_output, (word_hidden, word_cell_state) = self.word_RNN(word_embedding_seq,self.init_word_hidden())
            elif self.cell_type=='GRU':
                word_output, word_hidden = self.word_RNN(word_embedding_seq,self.init_word_hidden())
            if self.attn_act=='Tanh':
                word_annotation = torch.tanh(self.word_attn(word_output))
            elif self.attn_act=='ReLU':
                word_annotation = F.relu(self.word_attn(word_output))
            else:
                word_annotation = self.word_attn(word_output)
            word_attn = F.softmax(self.word_attn_combine(word_annotation),dim=0)
            word_attn_vec = self.attn_mul(word_output,word_attn) 

            da_input.append(word_attn_vec.view(self.hidden_dim*2, 1, -1))
        if len(da_input)==0:
            print("None of the words in this message do not exist in the given word embedding dict")
            return None
        da_input_mean = torch.stack(da_input).mean(dim=0)

        for i in range(len(empty_seq_da_idxs)):
            da_input.insert(i+empty_seq_da_idxs[i], da_input_mean)

        da_input = torch.cat(da_input).view(len(da_input), 1,-1)  # concat hidden vectors from the last cells of forward and backward LSTM
        da_input = F.dropout(da_input, p=self.dropout_rate, training=self.training)
        
        da_output, (da_hidden, da_cell_state) = self.da_RNN(da_input, self.init_da_hidden())
        relation_vec=torch.cat([da_output[arg1Idx],da_output[arg2Idx]]).view(1,-1)
        class_vec = self.das_to_class(relation_vec)
        type_vec = self.das_to_type(relation_vec)
        subtype_vec = self.da_to_subtype(da_output[arg2Idx])
        

        return class_vec.view(4),type_vec.view(16),subtype_vec.view(25),relation_vec

    
    ## Remove history of the hidden vector from the last instance
    def init_word_hidden(self):
        if self.is_cuda:
            if self.cell_type=='LSTM':
                return (
                    autograd.Variable(
                        torch.zeros(2, 1, self.hidden_dim)).cuda(),
                    autograd.Variable(
                        torch.zeros(2, 1, self.hidden_dim)).cuda())
            elif self.cell_type=='GRU':
                return autograd.Variable(
                        torch.zeros(2, 1, self.hidden_dim)).cuda()
        else:
            if self.cell_type=='LSTM':
                return (
                    autograd.Variable(
                        torch.zeros(2, 1, self.hidden_dim)),
                    autograd.Variable(
                        torch.zeros(2, 1, self.hidden_dim)))
            elif self.cell_type=='GRU':
                return autograd.Variable(
                        torch.zeros(2, 1, self.hidden_dim))

    def init_da_hidden(self):
        if self.is_cuda:
            if self.cell_type=='LSTM':
                return (
                    autograd.Variable(
                        torch.zeros(self.num_layer * self.num_direction, 1, self.hidden_dim)).cuda(),
                    autograd.Variable(
                        torch.zeros(self.num_layer * self.num_direction, 1, self.hidden_dim)).cuda())
            elif self.cell_type=='GRU':
                return autograd.Variable(
                        torch.zeros(self.num_layer * self.num_direction, 1, self.hidden_dim)).cuda()
        else:
            if self.cell_type=='LSTM':
                return (
                    autograd.Variable(
                        torch.zeros(self.num_layer * self.num_direction, 1, self.hidden_dim)),
                    autograd.Variable(
                        torch.zeros(self.num_layer * self.num_direction, 1, self.hidden_dim)))
            elif self.cell_type=='GRU':
                return autograd.Variable(
                        torch.zeros(self.num_layer * self.num_direction, 1, self.hidden_dim))



   


    


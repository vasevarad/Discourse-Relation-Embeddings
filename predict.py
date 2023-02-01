import discourseParsing.DiscourseParser as DP
import discourseParsing.utils.SenseLabeller as SL
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import csv
import pickle
import time
import math
import numpy as np
import argparse

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

parser = argparse.ArgumentParser(description='Discourse Parser Training')
def main():

    parser.add_argument('--input_dim', type=int, default=25,
                        help='the dimension of the hidden layer to be used. Default=25')
    parser.add_argument('--hidden_dim', type=int, default=25,
                        help='the dimension of the hidden layer to be used. Default=25')
    parser.add_argument('--seed', type=int, default=1, help='random seed to use. Default=1')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout Rate. Default=0.3')
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda?')
    parser.add_argument('--grad', type=str, default='SGD', help='Optimzer type: SGD? Adam? Default=SGD')
    parser.add_argument('--mini_batch', type=int, default=None, help='Optimzer type: SGD? Adam? Default=SGD')


    parser.add_argument('--model', type=str, default='./Trained_Models/',
                        help='pretrained model')
    parser.add_argument('--word_embedding_dict', type=str, default='/data/glove/glove_25.dict',
                        help='the path for language dict models')

    parser.add_argument('--test_file', type=str, help='test file')
    parser.add_argument('--test_id_file', type=str, default=None, help='test file')

    parser.add_argument('--split_one_arg', action='store_true', default=False, help='test file')

    parser.add_argument('--num_direction', type=int, default=2,
                        help='# of direction of RNN for sentiment detection, Default=2 (bidirectional)')
    parser.add_argument('--num_layer', type=int, default=1, help='# of direction of RNN layers')

    parser.add_argument('--cell_type', type=str, default='LSTM', help='cell selection: LSTM / GRU')

    parser.add_argument('--attn_act', type=str, default='None', help='Attention Activation Selection: None / Tanh / ReLU')

    parser.add_argument('--attn_type', type=str, default='element-wise',
                        help='Attention Aggregation Method: element-wise / vector-wise')



    # Parsing the arguments from the command
    opt = parser.parse_args()
    print(opt)
    input_dim=opt.input_dim
    hidden_dim=opt.hidden_dim
    num_direction=opt.num_direction
    dropout_rate=opt.dropout
    relDump={}
    msgAVGDump={}
    skips=[]



    is_cuda= opt.cuda
    sl=SL.SenseLabeller()
    
    # fix the seed as '1' for now
    torch.manual_seed(opt.seed)
    if is_cuda:
        torch.cuda.manual_seed(opt.seed)


    test_word_seqs=pickle.load(open(opt.test_file,"rb"))



    input_dim=opt.input_dim

    model = DP.DiscourseParser(opt)
    model.load_state_dict(torch.load(opt.model))
    if not is_cuda:
        model.to(torch.device("cpu"))
    

    loss_function = nn.BCEWithLogitsLoss()
    # Choose the gradient descent: SGD or Adam

    if is_cuda:
        model = model.cuda()
        loss_function = loss_function.cuda()

    total_start=time.time()


    with torch.no_grad():
        if opt.test_id_file:
            test_ids=[]
            with open(opt.test_id_file, 'r') as test_id_file:
                for line in test_id_file:
                    test_ids.append(int(line))
        else:
            if type(test_word_seqs) is list:
                test_ids=range(len(test_word_seqs))
            elif type(test_word_seqs) is dict:
                test_ids=list(test_word_seqs.keys())
        total_start = time.time()

        model.eval()
        start = time.time()
        for i in test_ids:
            if len(test_word_seqs[i]) < 2:
                if opt.split_one_arg:
                    half=int(len(test_word_seqs[i][0])/2)
                    splitted=[test_word_seqs[i][0][:half],test_word_seqs[i][0][half:]]
                    results = model(('Eval', 'N/A', 0, 1), splitted)
                    if results is None:
                        print("The message at '%d' doesn't have any word in the given embedding dict. Skipping this message"%(i))
                        continue
                    else:
                        class_vec, type_vec, subtype_vec, relation_vec=results
                    relDump[i]={0:torch.cat([class_vec, type_vec, subtype_vec, relation_vec.view(opt.hidden_dim*opt.num_direction*2)]).view(1, -1).cpu().numpy()}
                    msgAVGDump[i]=np.mean(list(relDump[i].values()),axis=0)
                    skips.append(i)
                    continue

                else:
                    skips.append(i)
                    continue
            for j in range(len(test_word_seqs[i])-1):
                results = model(('Eval', 'N/A', j, j+1), test_word_seqs[i])
                if results is None:
                    print("The message at '%d' doesn't have any word in the given embedding dict. Skipping this message"%(i))
                    continue
                else:
                    class_vec, type_vec, subtype_vec, relation_vec=results
                try:
                    relDump[i][j]=torch.cat([class_vec, type_vec, subtype_vec, relation_vec.view(opt.hidden_dim*opt.num_direction*2)]).view(1,-1).cpu().numpy()
                except KeyError:
                    relDump[i]={j:torch.cat([class_vec, type_vec, subtype_vec, relation_vec.view(opt.hidden_dim*opt.num_direction*2)]).view(1, -1).cpu().numpy()}
            if results is not None:
                msgAVGDump[i]=np.mean(list(relDump[i].values()),axis=0)
        end_time = timeSince(start)
        print("Done.")
        if opt.split_one_arg:
            print("Saved at: %s" % ('relDump_SOA_'+opt.model.split('/')[-1]+opt.test_file.split('/')[-1]+'.dict'))
        else:
            print("Saved at: %s" % ('relDump_'+opt.model.split('/')[-1]+opt.test_file.split('/')[-1]+'.dict'))

        print("Prediction Time: %s" % (end_time))
        if len(skips) > 0:
            if opt.split_one_arg:
                print("Warning %i messages were splitted to half because they didn't have more than one discourse argument"%(len(skips)))
            else:
                print("Warning %i messages were skipped because they didn't have more than one discourse argument"%(len(skips)))

            skip_file=open('skipped_or_splitted_ids_'+opt.model.split('/')[-1]+opt.test_file.split('/')[-1]+'.csv','w')
            for idx in skips:
                skip_file.write(str(idx)+'\n')
            skip_file.close()

        if opt.split_one_arg:
            pickle.dump(relDump,open('relDump_SOA_'+opt.model.split('/')[-1]+opt.test_file.split('/')[-1]+'.dict','wb'))
            pickle.dump(msgAVGDump, open('avgDump_SOA_' + opt.model.split('/')[-1] + opt.test_file.split('/')[-1] + '.dict', 'wb'))

        else:
            pickle.dump(relDump,open('relDump_'+opt.model.split('/')[-1]+opt.test_file.split('/')[-1]+'.dict','wb'))
            pickle.dump(msgAVGDump, open('avgDump_' + opt.model.split('/')[-1] + opt.test_file.split('/')[-1] + '.dict', 'wb'))


                

if __name__=="__main__":
    main()

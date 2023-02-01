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



def iteration(is_cuda, msg_id, word_seqs, tr_meta, tr_inst, tr_imipl, loss_fn, optim, model, training):
    for trainer_id, case, disCon, arg1Idx, arg2Idx in tr_meta[msg_id]:
        
        explicitInstances=tr_inst[trainer_id]['explicit']
        implicitInstances=tr_inst[trainer_id]['implicit']
        # explicit relation training
        explicit_loss = None
        implicit_loss = None
        explicit_losses_i=0
        implicit_losses_i=0
        if explicitInstances:
            model.zero_grad()
            class_vec, type_vec, subtype_vec, relation_vec = model((case, disCon, arg1Idx, arg2Idx),word_seqs[msg_id])
            for label, weight in explicitInstances['class']:
                if is_cuda:
                    label = label.cuda()
                loss = weight * loss_fn(class_vec, label)
                explicit_losses_i += float(loss)
                if explicit_loss:
                    explicit_loss += loss
                else:
                    explicit_loss = loss

            for label, weight in explicitInstances['type']:
                if is_cuda:
                    label = label.cuda()
                loss = weight * loss_fn(type_vec, label)
                explicit_losses_i += float(loss)
                if explicit_loss:
                    explicit_loss += loss
                else:
                    explicit_loss = loss

            for label, weight in explicitInstances['subtype']:
                if is_cuda:
                    label = label.cuda()
                loss = weight * loss_fn(subtype_vec, label)
                explicit_losses_i += float(loss)
                if explicit_loss:
                    explicit_loss += loss
                else:
                    explicit_loss = loss
            if training:
                explicit_loss.backward()
                optim.step()

        # implicit training

        if implicitInstances:
            model.zero_grad()
            class_vec, type_vec, subtype_vec, relation_vec = model((case, disCon, arg1Idx, arg2Idx),tr_imipl[trainer_id])
            for label, weight in implicitInstances['class']:
                if is_cuda:
                    label = label.cuda()
                loss = weight * loss_fn(class_vec, label)
                implicit_losses_i += float(loss)
                if implicit_loss:
                    implicit_loss += loss
                else:
                    implicit_loss = loss

            for label, weight in implicitInstances['type']:
                if is_cuda:
                    label = label.cuda()
                loss = weight * loss_fn(type_vec, label)
                implicit_losses_i += float(loss)
                if implicit_loss:
                    implicit_loss += loss
                else:
                    implicit_loss = loss

            for label, weight in implicitInstances['subtype']:
                if is_cuda:
                    label = label.cuda()
                loss = weight * loss_fn(subtype_vec, label)
                implicit_losses_i += float(loss)
                if implicit_loss:
                    implicit_loss += loss
                else:
                    implicit_loss = loss
            if training:
                implicit_loss.backward()
                optim.step()

        return explicit_losses_i, implicit_losses_i


parser = argparse.ArgumentParser(description='Discourse Parser Training')
def main():

    parser.add_argument('--input_dim', type=int, default=25,
                        help='the dimension of the hidden layer to be used. Default=25')
    parser.add_argument('--hidden_dim', type=int, default=25,
                        help='the dimension of the hidden layer to be used. Default=25')
    parser.add_argument('--seed', type=int, default=1, help='random seed to use. Default=1')
    parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout Rate. Default=0.3')
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda?')
    parser.add_argument('--grad', type=str, default='SGD', help='Optimzer type: SGD? Adam? Default=SGD')
    parser.add_argument('--mini_batch', type=int, default=None, help='Optimzer type: SGD? Adam? Default=SGD')


    parser.add_argument('--model_path', type=str, default='./Trained_Models/',
                        help='the path for saving intermediate models')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='load a pretrained model and train more. Default=None')
    parser.add_argument('--word_embedding_dict', type=str, default='/data/glove/glove_25.dict',
                        help='the path for language dict models')
    parser.add_argument('--tr_file', type=str,
                        help='training file')
    parser.add_argument('--tr_meta_file', type=str,
                        help='training meta file')
    parser.add_argument('--tr_meta_id_file', type=str, default=None,
                        help='train meta id file. Default=None')
    
    
    parser.add_argument('--train_shuffle', type=str, default='no',
                        help='no/replace/shuffle')
   
    parser.add_argument('--num_direction', type=int, default=2,
                        help='# of direction of RNN for sentiment detection, Default=2 (bidirectional)')
    parser.add_argument('--num_layer', type=int, default=1,
                        help='# of direction of RNN layers')


    
    parser.add_argument('--cell_type', type=str, default='LSTM',
                        help='cell selection: LSTM / GRU')

    parser.add_argument('--attn_act', type=str, default='None',
                        help='Attention Activation Selection: None / Tanh / ReLU')

    parser.add_argument('--attn_type', type=str, default='element-wise',
                        help='Attention Aggregation Method: element-wise / vector-wise')

    parser.add_argument('--valid_perc', type=float, default=0.1,
                        help='Setting aside the validation set. Default=10%')




    # Parsing the arguments from the command
    opt = parser.parse_args()
    print(opt)
    input_dim=opt.input_dim
    hidden_dim=opt.hidden_dim
    learning_rate=opt.lr
    optimizer_type=opt.grad
    num_direction=opt.num_direction
    dropout_rate=opt.dropout



    is_cuda= opt.cuda
    sl=SL.SenseLabeller()
    
    # fix the seed as '1' for now
    torch.manual_seed(opt.seed)
    if is_cuda:
        torch.cuda.manual_seed(opt.seed)


    word_seqs_orig_train=pickle.load(open(opt.tr_file,"rb"))

    meta_orig_train=pickle.load(open(opt.tr_meta_file,"rb"))
    meta_orig_trI_train=pickle.load(open(opt.tr_meta_file.replace('.odict','_labels.dict'),"rb"))
    meta_orig_imp_WS_train = pickle.load(open(opt.tr_meta_file.replace('.odict', '_implicit_word_seqs.dict'), "rb"))
    if opt.tr_meta_id_file:
        orig_train_ids=pickle.load(open(opt.tr_meta_id_file,'rb'))
    else:
        orig_train_ids=list(meta_orig_train.keys())

    # Separate validation set
    # final training
    train_end_idx=int((1-opt.valid_perc)*len(meta_orig_train))
    train_ids=orig_train_ids[:train_end_idx]

    # validation
    dev_ids=orig_train_ids[train_end_idx:]
    

    input_dim=opt.input_dim
    train_size=len(train_ids)
    valid_size=len(dev_ids)

    model_name=str(input_dim)+'_'+str(hidden_dim)+'_lr'+str(learning_rate).replace('.','_')+'_'+opt.cell_type+'_attnAct_'+opt.attn_act+'_'+optimizer_type+'_'
    print('# of messages (tr/dev)',train_size,valid_size)

    model = DP.DiscourseParser(opt)
    if opt.pretrained:
        model.load_state_dict(torch.load(opt.pretrained))

    loss_function = nn.BCEWithLogitsLoss()
    # Choose the gradient descent: SGD or Adam
    if optimizer_type=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if is_cuda:
        model = model.cuda()
        loss_function = loss_function.cuda()


    lowest_dev_loss= float('inf')
    # Training
    total_start=time.time()
    for epoch in range(1,opt.nEpochs+1):
        model.train()
        explicit_losses=0
        implicit_losses=0

        if opt.train_shuffle == 'replace':
            training_set = np.random.choice(train_ids, train_size, replace=True)
        elif opt.train_shuffle == 'shuffle':
            training_set = np.random.choice(train_ids, train_size, replace=False)
        else:
            training_set = train_ids
        if opt.mini_batch:
            training_set=np.random.choice(training_set,opt.mini_batch, replace=False)
        start=time.time()

        for i in training_set:
            # iteration(is_cuda, msg_id, word_seqs, tr_meta, tr_inst, tr_imipl, loss_fn, optim, model, training):
            explicit_losses_i,implicit_losses_i=iteration(is_cuda, i, word_seqs_orig_train, meta_orig_train, meta_orig_trI_train, meta_orig_imp_WS_train, loss_function, optimizer, model, True)
            explicit_losses+=explicit_losses_i
            implicit_losses+=implicit_losses_i

        explicit_losses=explicit_losses/len(training_set)
        implicit_losses=implicit_losses/len(training_set)
        end_time=timeSince(start)
        total_time=timeSince(total_start)
        print('[',"Epoch #: " + str(epoch),']')
        print("Training Loss: " + str((explicit_losses+implicit_losses)))
        print("Training Explicit Relation Loss:",explicit_losses)
        print("Training Implicit Relation Loss:", implicit_losses)
        print("Epoch Time: %s"%(end_time))
        print("Total Time: %s"%(total_time))

        # Check Test Performance every 10th iteration
        if epoch%10==0:
            model.eval()
            dev_explicit_losses=0
            dev_implicit_losses=0

            with torch.no_grad():

                for i in dev_ids:
                    explicit_losses_i, implicit_losses_i = iteration(is_cuda, i, word_seqs_orig_train, meta_orig_train,
                                                                     meta_orig_trI_train, meta_orig_imp_WS_train,
                                                                     loss_function, optimizer, model, False)
                    dev_explicit_losses += explicit_losses_i
                    dev_implicit_losses += implicit_losses_i

                dev_explicit_losses=dev_explicit_losses/len(dev_ids)
                dev_implicit_losses=dev_implicit_losses/len(dev_ids)
                dev_losses = dev_explicit_losses + dev_implicit_losses

                print('[', "Epoch #: " + str(epoch), '(Validation)]')
                print("Dev Loss: " + str(dev_losses))
                print("Dev Explicit Relation Loss:", dev_explicit_losses)
                print("Dev Implicit Relation Loss:", dev_implicit_losses)

            if dev_losses<lowest_dev_loss:
                lowest_dev_loss=dev_losses
                print("This is the best model up to this point: " + str(dev_losses))
                torch.save(model.state_dict(), opt.model_path + model_name + "dir_" + str(num_direction) + "_devLoss_"+ str(dev_losses)[:7] +'_epoch_' + str(epoch)+'_Dropout_'+str(dropout_rate).replace('.','_')+"_early_stop_saved.ptstdict")

                

if __name__=="__main__":
    main()

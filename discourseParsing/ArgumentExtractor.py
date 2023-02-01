import csv
import sys
import utils.PDTBAppendixSenses as PDTBas

# Find the disocurse connectives with the given length, extract discoruse arguments using them
def disconMarking(arg, forms, u_poss, head_idxs, connective_list,disConLength):
    prev_p_marker=arg[0]
    mwc_found=False
    i=arg[0]



    while i < arg[1]-disConLength: # the range should be (start, end-3) because 'on the one hand' + Parg2
        nWords=' '.join([ str(forms[j]) for j in range(i,i+disConLength)])

        if nWords in connective_list:
            if ('L' in u_poss[prev_p_marker:i] or 'M' in u_poss[prev_p_marker:i] or 'V' in u_poss[prev_p_marker:i] \
                or 'Y' in u_poss[prev_p_marker:i] or 'G' in u_poss[prev_p_marker:i]) \
                    and ('L' in u_poss[i+disConLength:arg[1]] or 'M' in u_poss[i+disConLength:arg[1]] or 'V' in u_poss[i+disConLength:arg[1]] \
                         or 'Y' in u_poss[i+disConLength:arg[1]] or 'G' in u_poss[i+disConLength:arg[1]]):
                return (('PArgCase2',nWords,(prev_p_marker,i),(i,arg[1])))

        i+=1
    return None


def parg_extraction(arg, forms, u_poss, head_idxs, connective_list):


    # Case 1: Beginning with discourse connectives:

    for disConLength in range(5,0,-1):
        nWords = ' '.join([str(form) for form in forms[arg[0]:arg[0]+disConLength]])

        startIdx=arg[0] + disConLength
        i = startIdx+1

        if nWords in connective_list:
            while i < arg[1]:
                if u_poss[i]==',' and ('L' in u_poss[startIdx:i] or 'M' in u_poss[startIdx:i] or 'V' in u_poss[startIdx:i] \
                or 'Y' in u_poss[startIdx:i] or 'G' in u_poss[startIdx:i]) \
                and ('L' in u_poss[i+1:arg[1]] or 'M' in u_poss[i+1:arg[1]] or 'V' in u_poss[i+1:arg[1]] \
                or 'Y' in u_poss[i+1:arg[1]] or 'G' in u_poss[i+1:arg[1]]):
                    return (('PArgCase1',nWords,(arg[0], i),(i,arg[1])))

                i += 1

    # Case 2: Discourse connectives in the middle:
    # multi-word connectives filtering
    # finding 4-word length connectives (the longest PDTB connective)
    prev_s_marker = arg[0]
    i=arg[0]



    # finding 5-word to 1-word length connectives
    for disConLength in range(5,0,-1):
        temp_pargs = disconMarking(arg, forms, u_poss, head_idxs, connective_list, disConLength)
        if temp_pargs:
            return temp_pargs


    # tweet-specific '&' tags

    i = arg[0]
    while i < arg[1]:
        if u_poss[i] =='&' or u_poss[i] =='P' or (forms[i]==',' and u_poss[i]==','):
            if ('L' in u_poss[arg[0]:i] or 'M' in u_poss[arg[0]:i] or 'V' in u_poss[arg[0]:i] \
                or 'Y' in u_poss[arg[0]:i] or 'G' in u_poss[arg[0]:i]) \
                and ('L' in u_poss[i:arg[1]] or 'M' in u_poss[i:arg[1]] or 'V' in u_poss[i:arg[1]] \
                or 'Y' in u_poss[i:arg[1]] or 'G' in u_poss[i:arg[1]]):
                return (('PArgCase2',forms[i],(arg[0],i),(i, arg[1])))
        i+=1

    return ('SArg',arg)

def arg_extraction(idxs, forms, u_poss, head_idxs, connective_list):
    # Sentence extraction
    s_markers = []  # indexes for boudnaries sentences
    twt_len=len(idxs)
    prev_s_marker=0
    for i in range(len(idxs)):
        if u_poss[i] == 'E':
            if i!=0:
                s_markers.append(i)
            if i!=twt_len-1:
                s_markers.append(i + 1)
            # base case E on the first idx or the last idx
            prev_s_marker=i+1
        elif u_poss[i] == ',' and (len(forms[i])>1 or forms[i]=='.' or forms[i]=='!'): # take '.' or '!!!' as a sentence demarcator
            if prev_s_marker==i: # append this this token to the previous sentence if this is a consecutive marker
                if len(s_markers)==0:
                    s_markers.append(i+1)
                else:
                    s_markers[len(s_markers)-1]=i+1
            else:
                s_markers.append(i + 1)
            prev_s_marker = i + 1
    sarg_idxs=[]
    
    prev_s_marker=0
    for i in s_markers:
        sarg_idxs.append((prev_s_marker,i)) #(start index, end index+1)
        prev_s_marker=i
    if len(s_markers)==0: # when there are no sentences in a tweet (one-sentence tweet)
        sarg_idxs.append((0, twt_len)) # make it as one sentence arg
    else:
        if sarg_idxs[len(sarg_idxs)-1][1] != twt_len: # if the parser didn't capture the last senten #sarg_idxs=[sarg for sarg in sarg_idxs if sarg[0] == sarg[1]] # remove empty dus which fall into the base cases
            sarg_idxs.append((sarg_idxs[len(sarg_idxs) - 1][1], twt_len))
    final_arg_list=[]

    for sarg in sarg_idxs:
        if sarg[0] == sarg[1]: #edge cases where sentence marker 'E' and '.' was overlapped
            continue
        arg=parg_extraction(sarg, forms, u_poss, head_idxs, connective_list)
        final_arg_list.append(arg)

    return final_arg_list

def argWriting(prev_end_idx, argIdx, arg_start_idxs, arg_end_idxs, forms, u_poss, output_csv, output_a_pos_csv, twt_id, arg_id, disconnect_found, dup_found, argType):

    if prev_end_idx != argIdx[1]:
        disconnect_found = True
    prev_end_idx = argIdx[1]

    if argIdx[0] in arg_start_idxs:
        dup_found = True
    arg_start_idxs.add(argIdx[0])

    if argIdx[1] in arg_end_idxs:
        dup_found = True
    arg_end_idxs.add(argIdx[1])
    arg_form_str = ''
    pos_str = ''

    for j in range(argIdx[0], argIdx[1]):
        arg_form_str += ' ' + str(forms[j])
        pos_str += ' ' + (u_poss[j])
    output_csv.writerow([twt_id, argType, j, arg_id, arg_form_str[1:]])
    output_a_pos_csv.writerow([twt_id, arg_id, pos_str[1:]])

def main():
    tweebo_file=open(sys.argv[1],'r')
    socialMedia=sys.argv[2]
    expConns = [u'' + discon for discon in PDTBas.explicitConnectiveRelationDict.keys()]
    impConns = [u'' + discon for discon in PDTBas.implicitConnectiveRelationDict.keys()]
    connective_list=expConns+impConns
    # Max Length of discourse connectives=5

    if socialMedia=='y':
        raise Exception("To be implemented")
        #connective_list=[include all brown clusters of discourse connectives!]


    #id, form, lemma, UPOS, XPOS, FEATS, head, dependency relation
    #20      Heartthrob      _       ^       ^       _       21      _


    output_file=open(sys.argv[1]+'_args.csv','w')
    output_trainer_file = open(sys.argv[1] + '_args_train_meta.csv', 'w')
    output_a_pos_file=open(sys.argv[1]+'_a_pos.tsv','w')
    output_t_pos_file=open(sys.argv[1]+'_t_pos.tsv','w')
    error_file=open(sys.argv[1]+'_dup_errors.txt','w')
    output_csv=csv.writer(output_file)
    output_trainer_csv = csv.writer(output_trainer_file)
    output_a_pos_csv=csv.writer(output_a_pos_file,delimiter='\t')
    output_t_pos_csv=csv.writer(output_t_pos_file,delimiter='\t')

    twt_id=1

    output_csv.writerow(['tweet_id','arg_type','arg_end_idx','arg_id','message'])
    output_trainer_csv.writerow(['tweet_id','trainer_id','Case_Type','Discourse_Connective','Arg1_id','Arg2_id'])
    error_file.write('message_id,dup_error,disconnect_error,arg_idxs\n')

    sentences=[]
    idxs=[]
    forms=[]
    u_poss=[]
    head_idxs=[]
    t_pos=[]
    a_pos=[]
    for line in tweebo_file:
        if line=='\n': # new line detected: new tweet parsing result
            # (1) sentence extraction: segmenting with ',', 'E'
            if socialMedia=='n':
                twt_arg_tuples=arg_extraction(idxs, forms, u_poss, head_idxs, connective_list)
            else:
                raise Exception("To be implemented")
            u_pos_str=''
            for u_pos in u_poss:
                u_pos_str+=' '+u_pos

            #tester_code (1) if arg idxs are always connected, (2) if there are duplicates
            arg_start_idxs=set()
            arg_end_idxs=set()
            dup_found=False
            disconnect_found=False
            prev_end_idx=0
            argCnt=0
            trainerCnt=0
            for argTuple in twt_arg_tuples:
                if argTuple[0]=='SArg':
                    argIdx=argTuple[1]
                    arg_id = str(twt_id) + '-' + str(argCnt)
                    argCnt+=1
                    # If the previuos ending index is not the start of the current arg start index, disconnection is found!
                    argWriting(prev_end_idx, argIdx, arg_start_idxs, arg_end_idxs, forms, u_poss, output_csv,output_a_pos_csv, twt_id, arg_id,disconnect_found,dup_found,'SArg')


                else:
                    disConWord = argTuple[1]

                    argIdx1 = argTuple[2]
                    arg_id1 = str(twt_id) + '-' + str(argCnt)
                    argCnt+=1
                    argIdx2 = argTuple[3]
                    arg_id2 = str(twt_id) + '-' + str(argCnt)
                    argCnt += 1

                    argWriting(prev_end_idx, argIdx1, arg_start_idxs, arg_end_idxs, forms, u_poss, output_csv,output_a_pos_csv, twt_id, arg_id1, disconnect_found, dup_found, 'PArg')
                    argWriting(prev_end_idx, argIdx2, arg_start_idxs, arg_end_idxs, forms, u_poss, output_csv,output_a_pos_csv, twt_id, arg_id2, disconnect_found, dup_found, 'PArg')

                    if argTuple[0] == 'PArgCase1':  # first arg is Arg 2, second arg is Arg 1
                        output_trainer_csv.writerow([twt_id,str(twt_id) + '-' + str(trainerCnt), 'Case1', disConWord, arg_id2, arg_id1])
                        trainerCnt+=1

                    else:
                        output_trainer_csv.writerow([twt_id, str(twt_id) + '-' + str(trainerCnt), 'Case2', disConWord, arg_id1, arg_id2])
                        trainerCnt+=1


            output_t_pos_csv.writerow([twt_id,u_pos_str[1:]])
            if dup_found or disconnect_found:
                #'message_id,dup_error,discon_error,arg_idxs'
                error_line=str(twt_id)+','
                if dup_found:
                    error_line+='1,'
                else:
                    error_line+='0,'
                if disconnect_found:
                    error_line+='1,'
                else:
                    error_line+='0,'
                error_line+=str(twt_arg_tuples)+'\n'
                error_file.write(error_line)

            #after extraction,  increment twt_id
            twt_id+=1
            idxs = []
            forms = []
            u_poss = []
            head_idxs = []
            sentences=[]
            s_ptr=0 # index after period

        else:
            tokens=line.split('\t')
            idxs.append(int(tokens[0]))
            forms.append(tokens[1])
            lemma=tokens[2]
            u_poss.append(tokens[3])
            s_pos=tokens[4]
            feat=tokens[5]
            head_idxs.append(int(tokens[6]))
            dep_rel=tokens[7]

    tweebo_file.close()
    output_file.close()
    output_trainer_file.close()
    output_a_pos_file.close()
    output_t_pos_file.close()
    error_file.close()



if __name__=='__main__':

    if len(sys.argv) != 3:
        print("USAGE> crt_arg_extractor.py [tweebo_output.predict] [social_media_extraction? (y/n)]")
        sys.exit()
    main()


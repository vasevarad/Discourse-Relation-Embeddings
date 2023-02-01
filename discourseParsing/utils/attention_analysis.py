import PDTBAppendixSenses as PDTBas
import sys
import numpy as np


if len(sys.argv)!=2:
    print('USAGE> python attention_analysis.py [word attention dict file]')
    sys.exit()


explicitDict=PDTBas.explicitConnectiveRelationDict
implicitDict=PDTBas.implicitConnectiveRelationDict


wordAvgAttn[word]=float(np.mean(wordAttnDump[word]))


print('Discourse Connective attention:')
print('Non-Discourse Connective attention:')


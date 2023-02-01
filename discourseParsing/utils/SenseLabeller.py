try:
    import PDTBAppendixSenses as PDTBas
except ModuleNotFoundError:
    from . import PDTBAppendixSenses as PDTBas
import numpy as np
class SenseLabeller:
    class SenseNode:
        def __init__(self,sense,level,parent,children):
            self.sense=sense
            self.level=level
            self.parent=parent
            self.children=children
            
    def __init__(self):
        self.senseIdx, self.senseMap=self.senseInit()
        self.weightDict=self.weightInit()

    def senseInit(self):
        # TEMPORAL Relations
        # Class
        temporal=self.SenseNode('TEMPORAL',1,None,{})
        # Types
        asynchronous=self.SenseNode('Asynchronous',2,temporal,None)
        synchrony=self.SenseNode('Synchrony',2,temporal,{})
        temporal.children['Asynchronous']=asynchronous
        temporal.children['Synchrony']=synchrony
        # Subtypes
        precedence=self.SenseNode('precedence',3,synchrony,None)
        succession=self.SenseNode('succession',3,synchrony,None)
        synchrony.children['precedence']=precedence
        synchrony.children['succession']=succession

        # COMPARISON Relations
        # Class
        comparison=self.SenseNode('COMPARISON',1,None,{})
        # Types
        contrast=self.SenseNode('Contrast',2,comparison,{})
        pragmaticContrast=self.SenseNode('Pragmatic_Contrast',2,comparison,None)
        concession=self.SenseNode('Concession',2,comparison,{})
        pragmaticConcession=self.SenseNode('Pragmatic_Concession',2,comparison,None)
        comparison.children['Contrast']=contrast
        comparison.children['Pragmatic_Contrast']=pragmaticContrast
        comparison.children['Concession']=concession
        comparison.children['Pragmatic_Concession']=pragmaticConcession
        # Subtypes
        juxtaposition=self.SenseNode('juxtaposition',3,contrast,None)
        opposition=self.SenseNode('opposition',3,contrast,None)
        expectation=self.SenseNode('expectation',3,concession,None)
        contraExpectation=self.SenseNode('contra-expectation',3,concession,None)
        contrast.children['juxtaposition']=juxtaposition
        contrast.children['opposition']=opposition
        concession.children['expectation']=expectation
        concession.children['contra-expectation']=contraExpectation
        
        # CONTINGENCY Relations
        # Class
        contingency=self.SenseNode('CONTINGENCY',1,None,{})
        # Types
        cause=self.SenseNode('Cause',2,contingency,{})
        pragmaticCause=self.SenseNode('Pragmatic_Cause',2,contingency,{})
        condition=self.SenseNode('Condition',2,contingency,{})
        pragmaticCondition=self.SenseNode('Pragmatic_Condition',2,contingency,{})
        contingency.children['Cause']=cause
        contingency.children['Pragmatic_Cause']=pragmaticCause
        contingency.children['Condition']=condition
        contingency.children['Pragmatic_Condition']=pragmaticCondition
        # Subtypes
        reason=self.SenseNode('reason',3,cause,None)
        result=self.SenseNode('result',3,cause,None)
        justification=self.SenseNode('justification',3,pragmaticCause,None)
        hypothetical=self.SenseNode('hypothetical',3,condition,None)
        general=self.SenseNode('general',3,condition,None)
        unrealPresent=self.SenseNode('unreal_present',3,condition,None)
        unrealPast=self.SenseNode('unreal_past',3,condition,None)
        factualPresent=self.SenseNode('factual_present',3,condition,None)
        factualPast=self.SenseNode('factual_past',3,condition,None)
        relevance=self.SenseNode('relevance',3,pragmaticCondition,None)
        implicitAssertion=self.SenseNode('implicit_assertion',3,pragmaticCondition,None)
        cause.children['reason']=reason
        cause.children['result']=result
        pragmaticCause.children['justification']=justification
        condition.children['hypothetical']=hypothetical
        condition.children['general']=general
        condition.children['unreal_present']=unrealPresent
        condition.children['unreal_past']=unrealPast
        condition.children['factual_present']=factualPresent
        condition.children['factual_past']=factualPast
        pragmaticCondition.children['relevance']=relevance
        pragmaticCondition.children['implicit_assertion']=implicitAssertion

        # EXPANSION Relations
        # Class
        expansion=self.SenseNode('EXPANSION',1,None,{})
        # Types
        conjunction=self.SenseNode('Conjunction',2,expansion,None)
        instantiation=self.SenseNode('Instantiation',2,expansion,None)
        restatement=self.SenseNode('Restatement',2,expansion,{})
        alternative=self.SenseNode('Alternative',2,expansion,{})
        exception=self.SenseNode('Exception',2,expansion,None)
        list_=self.SenseNode('List',2,expansion,None)
        expansion.children['Conjunction']=conjunction
        expansion.children['Instantiation']=instantiation
        expansion.children['Restatement']=restatement
        expansion.children['Alternative']=alternative
        expansion.children['Exception']=exception
        expansion.children['List']=list_
        # Subtypes
        specification=self.SenseNode('specification',3,restatement,None)
        equivalence=self.SenseNode('equivalence',3,restatement,None)
        generalization=self.SenseNode('generalization',3,restatement,None)
        conjunctive=self.SenseNode('conjunctive',3,alternative,None)
        disjunctive=self.SenseNode('disjunctive',3,alternative,None)
        chosenAlternative=self.SenseNode('chosen_alternative',3,alternative,None)
        restatement.children['specification']=specification
        restatement.children['equivalence']=equivalence
        restatement.children['generalization']=equivalence
        alternative.children['conjunctive']=conjunctive
        alternative.children['disjunctive']=disjunctive
        alternative.children['chosen_alternative']=chosenAlternative

        senseMap={'TEMPORAL':temporal, 'Asynchronous':asynchronous, 'Synchrony':synchrony, 'precedence':precedence, 'succession':succession, 'COMPARISON':comparison, 'Contrast':contrast, 'juxtaposition':juxtaposition, 'opposition':opposition, 'Pragmatic_Contrast':pragmaticContrast, 'Concession':concession, 'expectation':expectation, 'contra-expectation':contraExpectation, 'Pragmatic_Concession':pragmaticConcession, 'CONTINGENCY':contingency, 'Cause':cause, 'reason':reason, 'result':result, 'Pragmatic_Cause':pragmaticCause, 'justification':justification, 'Condition':condition, 'hypothetical':hypothetical, 'general':general, 'unreal_present':unrealPresent, 'unreal_past':unrealPast, 'factual_present':factualPresent, 'factual_past':factualPast, 'Pragmatic_Condition':pragmaticCondition, 'relevance':relevance, 'implicit_assertion':implicitAssertion, 'EXPANSION':expansion, 'Conjunction':conjunction, 'Instantiation':instantiation, 'Restatement':restatement, 'specification':specification, 'equivalence':equivalence, 'generalization':generalization, 'Alternative':alternative, 'conjunctive':conjunctive, 'disjunctive':disjunctive, 'chosen_alternative':chosenAlternative, 'Exception':exception, 'List':list_}
        senseIdx={'TEMPORAL':0, 'COMPARISON':1, 'CONTINGENCY':2, 'EXPANSION':3, 'Asynchronous':0, 'Synchrony':1, 'Contrast':2, 'Pragmatic_Contrast':3, 'Concession':4, 'Pragmatic_Concession':5, 'Cause':6, 'Pragmatic_Cause':7, 'Condition':8, 'Pragmatic_Condition':9, 'Conjunction':10, 'Instantiation':11, 'Restatement':12, 'Alternative':13, 'Exception':14, 'List':15, 'precedence':0, 'succession':1, 'juxtaposition':2, 'opposition':3, 'expectation':4, 'contra-expectation':5, 'reason':6, 'result':7, 'justification':8, 'altlex':9, 'hypothetical':10, 'general':11, 'unreal_present':12, 'unreal_past':13, 'factual_present':14, 'factual_past':15, 'relevance':16, 'implicit_assertion':17, 'specification':18, 'equivalence':19, 'generalization':20, 'conjunctive':21, 'disjunctive':22, 'chosen_alternative':23, 'entrel':24}
        return senseIdx, senseMap

    def weightInit(self):
        # return weights for class and type
        explicitWeights={}
        implicitWeights={}
        explicitDict=PDTBas.explicitConnectiveRelationDict
        implicitDict = PDTBas.implicitConnectiveRelationDict

        for disCon in explicitDict:
            classWeights = np.array([0] * 4)  # Four types of discourse relation classes
            typeWeights = np.array([0] * 16)  # 16 types of discourse relation types
            subtypeWeights = np.array([0] * 25) # 25 types of discourse relation subtypes

            for relation in explicitDict[disCon]:
                rels = relation.split('/')
                for rel in rels:
                    # alpha / beta update

                    if self.senseMap[rel].level == 3:
                        classWeights[self.senseIdx[self.senseMap[rel].parent.parent.sense]] += explicitDict[disCon][relation]
                        typeWeights[self.senseIdx[self.senseMap[rel].parent.sense]] += explicitDict[disCon][relation]
                        subtypeWeights[self.senseIdx[self.senseMap[rel].sense]] += explicitDict[disCon][relation]
                    elif self.senseMap[rel].level == 2:
                        classWeights[self.senseIdx[self.senseMap[rel].parent.sense]] += explicitDict[disCon][relation]
                        typeWeights[self.senseIdx[self.senseMap[rel].sense]] += explicitDict[disCon][relation]
                    else:
                        classWeights[self.senseIdx[self.senseMap[rel].sense]] += explicitDict[disCon][relation]

            if classWeights.sum() != 0:
                classWeights=classWeights/classWeights.sum()
            else:
                classWeights=None

            if typeWeights.sum() != 0:
                typeWeights=typeWeights/typeWeights.sum()
            else:
                typeWeights=None

            if subtypeWeights.sum() != 0:
                subtypeWeights=subtypeWeights/subtypeWeights.sum()
            else:
                subtypeWeights=None
            
            explicitWeights[disCon]={'class':classWeights,'type':typeWeights,'subtype':subtypeWeights}

        for disCon in implicitDict:
            classWeights = np.array([0] * 4)  # Four types of discourse relation classes
            typeWeights = np.array([0] * 16)  # 16 types of discourse relation types
            subtypeWeights = np.array([0] * 25) # 25 types of discourse relation subtypes
            
            for relation in implicitDict[disCon]:
                rels = relation.split('/')
                for rel in rels:
                    # alpha / beta update

                    if self.senseMap[rel].level == 3:
                        classWeights[self.senseIdx[self.senseMap[rel].parent.parent.sense]] += implicitDict[disCon][relation]
                        typeWeights[self.senseIdx[self.senseMap[rel].parent.sense]] += implicitDict[disCon][relation]
                        subtypeWeights[self.senseIdx[self.senseMap[rel].sense]] += implicitDict[disCon][relation]
                    elif self.senseMap[rel].level == 2:
                        classWeights[self.senseIdx[self.senseMap[rel].parent.sense]] += implicitDict[disCon][relation]
                        typeWeights[self.senseIdx[self.senseMap[rel].sense]] += implicitDict[disCon][relation]
                    else:
                        classWeights[self.senseIdx[self.senseMap[rel].sense]] += implicitDict[disCon][relation]

            if classWeights.sum() != 0:
                classWeights=classWeights/classWeights.sum()
            else:
                classWeights=None

            if typeWeights.sum() != 0:
                typeWeights=typeWeights/typeWeights.sum()
            else:
                typeWeights=None

            if subtypeWeights.sum() != 0:
                subtypeWeights=subtypeWeights/subtypeWeights.sum()
            else:
                subtypeWeights=None
            
            implicitWeights[disCon]={'class':classWeights,'type':typeWeights,'subtype':subtypeWeights}

        return {'explicit':explicitWeights,'implicit':implicitWeights}



        
        
        

#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-


import sys
#from LabelledTree import *


# B. Crabbé
#This implements the classical Magerman's head propagation tables
class PropagationTable:
    def __init__(self):
        self.dico = {}
        #marie : fast link from rule (lhs -> rhs) to head index in rhs
        self.rule_to_index = {}
        self.failing_rules = {}

    # head rules storage
    def add_mapping(self,key,value,direction):
        if not key in self.dico:
            self.dico[key] = []
        self.dico[key].append((value,direction))
           
    def add_alias(self,symbol,alias):
        for key in self.dico.keys():
            rules = self.dico[key]
            newrules = []
            for rule in rules:
                (r,dir) = rule
                newrule = []
                for elt in r:
                    if elt == symbol:
                        newrule.append(alias)
                    newrule.append(elt)
                newrules.append((newrule,dir))
            self.dico[key] = newrules
        if not alias in self.dico and symbol in self.dico:
            self.dico[alias] = self.dico[symbol]

    def add_aliases(self,symbol,aliaslist):
        for elt in aliaslist:
            self.add_alias(symbol,elt)

    # Marie 
    # returns the index of head child, according to propagation table
    # cf_rule is a tuple, first elt = lhs symbol, other elements = the rhs symbols
    def find_child_index(self, cf_rule):
        lhs = cf_rule[0]
        if lhs not in self.dico:
            return 
        h_rules = self.dico[lhs]
        for h_rule in h_rules:
            (priorities,d) = h_rule
            scan = cf_rule[1:]
            scanindex = range(len(scan))
            if d == "R":
                list(reversed(scanindex))
            for p in priorities:
                for index in scanindex:
                    #sys.stderr.write("comparing " + str(p) + 'and' + str(scan[index])+'\n')
                    if (p == scan[index][:len(p)] or p == '*'):
                        return index
        return None

    # Marie
    # returns the index of head child, according to propagation table
    # (searches for it in index dictionary, 
    #  or via find_child_index for first occurrence)
    def head_index(self,root,children):
        syms = tuple([root.label.split('-')[0]]+list(map(lambda x:x.label.split('##')[0],children)))
        #syms = tuple([root.label+list(map(lambda x:x.label.split('##')[0],children)))
        return self.get_child_index(syms)
    
    def get_child_index(self, cf_rule):
        #print(cf_rule)
        i = self.find_child_index(cf_rule)
        if i == None:
            return (len(cf_rule)-2)
        return i
        # if such rule was never encountered before
        #if not cf_rule in self.rule_to_index:
        #    i = self.find_child_index(cf_rule)
        #    if i == None:
        #        self.failing_rules[cf_rule] = self.failing_rules.get(cf_rule, 0) + 1
        #        return (len(cf_rule)-2)
        #    self.rule_to_index[cf_rule] = i
        #    return i
        # else, such rule already encountered -> return stored index
        #return self.rule_to_index[cf_rule]

    # statistics on head rule applications
    def stats(self):
        for rule in sorted(self.failing_rules.items(), lambda x,y: cmp(y[1], x[1]) or cmp(x[0], y[0])):
            print ("MISSING HEAD FOR: (" + str(rule[1]) + ') ' + cf_rule_tostring(rule[0]))

    def display_table(self):
        res = "----------------------------------------------------\n"
        for elt in sorted(self.dico.keys()):
            rules = self.dico[elt]
            for r in rules:
                (p,dir) = r
                res = res + elt+" ("+dir+") : "+" ".join(p) +"\n"
            res = res+"----------------------------------------------------\n"
        return res

#Dummy table for testing
def dummy_propag_table():
    ptable = PropagationTable()
    ptable.add_mapping("S",["VN"],"L")
    ptable.add_mapping("VN",["V"],"R")
    ptable.add_mapping("PP",["NP"],"L")
    ptable.add_mapping("NP",["N"],"L")
    ptable.add_aliases("V",["VINF","VFIN","VPP"])
    ptable.add_aliases("N",["NC","NPP"])
    ptable.add_aliases("S",["SENT"])
    return ptable
       
#Arun and Keller's table for treebank with symset 4 
def ak_sym4_table():
    ptable = PropagationTable()
    #A&K Table
    ptable.add_mapping("AP",["A","N","V"],"R")
    ptable.add_mapping("AdP",["ADV"],"R")
    ptable.add_mapping("AdP",["P","D","C"],"L")
    ptable.add_mapping("COORD",["C"],"L")
    ptable.add_mapping("COORD",["*"],"R")
    ptable.add_mapping("NP",["N","PRO","A","ADV"],"R")
    ptable.add_mapping("NP",["NP"],"L")
    ptable.add_mapping("NP",["*"],"R")
    # marie: inhib ben C : ptable.add_mapping("PP",["NP","N","V","ADV","A","CL","P"],"R")
    # original head rule from Arun (prep = head of PP)
    ptable.add_mapping("PP",['P', 'CL', 'A', 'ADV', 'V', 'N'], 'R')
    ptable.add_mapping("SENT",["VN","V","NP","Srel","Ssub","Sint"],"L")
    ptable.add_mapping("Sint",["VN","V"],"L")
    ptable.add_mapping("Sint",["*"],"R")
    ptable.add_mapping("Srel",["VN","V"],"L")
    ptable.add_mapping("Ssub",["VN","V"],"L")
    ptable.add_mapping("Ssub",["*"],"R")
    ptable.add_mapping("VN",["V"],"R")
    ptable.add_mapping("VPinf",["VN","V"],"L")
    ptable.add_mapping("VPinf",["*"],"R")
    ptable.add_mapping("VPpart",["VN","V"],"L")
    ptable.add_mapping("VPpart",["*"],"R")
    #Extension to symset 4
    ptable.add_aliases("V",["VIMP","VINF","VS","VPP","VPR"])
    ptable.add_aliases("N",["NC","NPP"])
    ptable.add_aliases("C",["CS","CC"])
    ptable.add_aliases("CL",["CLS","CLO","CLR"])
    ptable.add_aliases("P",["P+D","P+PRO"])
    ptable.add_aliases("A",["ADJ","ADJWH"])
    ptable.add_aliases("ADV",["ADVWH"])
    ptable.add_aliases("PRO",["PROWH","PROREL"])
    ptable.add_aliases("D",["DET","DETWH"])
    return ptable

#Modified Arun and Keller's table for treebank with symset 4 
# marie : modified to cope with MFT treebank as well
def sym4_table():
    ptable = PropagationTable()
    #A&K Table
    ptable.add_mapping("AP",["A","N","V"],"R")
    # marie "(AP (COORD (C sinon) (AP (A normaux))))
    ptable.add_mapping("AP",["COORD"],"L")
    # marie : specific to MFT
    ptable.add_mapping("AP",["PREF"],"L")
    ptable.add_mapping("AdP",["ADV"],"R")
    ptable.add_mapping("AdP",["P","P+D","D","C"],"L")
    # marie : specific to MFT
    ptable.add_mapping("AdP",["COORD"],"L")
    # mathieu : add PONCT exemple :
    # marie : to test this cf. sentence or vp coordination:
    # ptable.add_mapping("COORD",["VN", "V","C","PONCT"],"L")
    ptable.add_mapping("COORD",["C","PONCT"],"L")
    ptable.add_mapping("COORD",["*"],"R")
    # marie !! : pour le N c'est plutot le premier N à gauche
    # année 1999, mardi soir, filiale publicité, lendemain matin, case départ, volume record, plan média
    # "fin janvier" --> ?? contrexemple?
    # ==> crée pbs dans le cas de NP contenant appositions et mal reconnus
    # exemple : la Caixa, premiere caisse d'epargne espagnole
    # marie : P+PRO a une distribution de N
    # ptable.add_mapping("NP",["N","PRO","A","ADV"],"R")
    ptable.add_mapping("NP",["N","P+PRO"],"L")
    ptable.add_mapping("NP",["PRO","A","ADV"],"R")
    ptable.add_mapping("NP",["NP"],"L")
    ptable.add_mapping("NP",["*"],"R")
    # marie: inhib ben C : ptable.add_mapping("PP",["NP","N","V","ADV","A","CL","P"],"R")
    # original head rule from Arun (prep = head of PP)
    # ARUN : ptable.add_mapping("PP",['P', 'CL', 'A', 'ADV', 'V', 'N'], 'R')
    # marie : MODIFICATION OF ARUN : add PRO cf: (PP (PRO dont))
    #       MODIFICATION OF ARUN : add NP cf: (PP (NP (P+PRO auxquels)))
    ptable.add_mapping("PP",['P', 'P+D', 'CL', 'A', 'ADV', 'V', 'N', 'PRO', 'NP'], 'R')
    # marie : specific to MFT
    ptable.add_mapping("PP",['COORD'], 'L')
    # ARUN : ptable.add_mapping("SENT",["VN","V","NP","Srel","Ssub","Sint"],"L")
    # marie : MODIFICATION add COORD, VPinf, VPpart
    # mathieu : add I, AP
    ptable.add_mapping("SENT",["VN","V","NP","Srel","Ssub","Sint","VPinf","VPpart","COORD", "PP", "AdP", "AP", "I"],"L")
    ptable.add_mapping("Sint",["VN","V"],"L")
    ptable.add_mapping("Sint",["*"],"R")
#marie    ptable.add_mapping("Srel",["VN","V"],"L")
# fausses relatives dont = "dont, en france, FO et la CFDT.
# mathieu : tu as mis PP pourquoi? moi j'avais rajouté NP
    ptable.add_mapping("Srel",["VN","V","NP"],"L")
    # marie : specific to MFT
    ptable.add_mapping("Srel",["COORD"],"L")
    #mathieu : add "C" : pourquoi? exemple??
    #ptable.add_mapping("Ssub",["VN","V"],"L")
    ptable.add_mapping("Ssub",["VN","V", "C"],"L")
    ptable.add_mapping("Ssub",["*"],"R")
    ptable.add_mapping("VN",["V"],"R")
    ptable.add_mapping("VPinf",["VN","V"],"L")
    ptable.add_mapping("VPinf",["*"],"R")
    # mathieu : add AP : pourquoi? exemple?
    #ptable.add_mapping("VPpart",["VN","V"],"L")
    ptable.add_mapping("VPpart",["VN","V","AP"],"L")
    ptable.add_mapping("VPpart",["*"],"R")
    #Extension to symset 4
    ptable.add_aliases("V",["VIMP","VINF","VS","VPP","VPR"])
    # marie : added NNC (specific to MFT4)
    ptable.add_aliases("N",["NC","NPP","NNC"])
    ptable.add_aliases("C",["CS","CC"])
    ptable.add_aliases("CL",["CLS","CLO","CLR"])
    # marie : en fait P+PRO a une distribution de N (cf. est au sein du NP...)
    #ptable.add_aliases("P",["P+D","P+PRO"])
    # marie : inhibé car intégré directement supra    ptable.add_aliases("P",["P+D"])
    ptable.add_aliases("A",["ADJ","ADJWH"])
    ptable.add_aliases("ADV",["ADVWH"])
    ptable.add_aliases("PRO",["PROWH","PROREL"])
    ptable.add_aliases("D",["DET","DETWH"])

    # marie : specific to MFT (NC = non constituent coordinate)
    # -> on prend simplement le premier constituant...
    ptable.add_mapping("NC",["*"],"L")
    #        VP present dans le MFT (en cas de coord de VP)
    ptable.add_mapping("VP",["VN"],"L")
    return ptable

def cohead_table():
    ptable = PropagationTable()
    ptable.add_mapping("PP",["P","CL","A","ADV","V","N","NP"],"R")
    ptable.add_aliases("V",["VPP","VINF","VS","VIMP","VPR"])
    ptable.add_aliases("N",["NC","NPP"])
    ptable.add_aliases("C",["CS","CC"])
    ptable.add_aliases("CL",["CLS","CLO","CLR"])
    ptable.add_aliases("P",["P+D","P+PRO"])
    ptable.add_aliases("A",["ADJ","ADJWH"])
    ptable.add_aliases("ADV",["ADVWH"])
    ptable.add_aliases("PRO",["PROWH","PROREL"])
    ptable.add_aliases("D",["DET","DETWH"])
    return ptable

#++++++++++++++++
# propag tables derived from Ane's work

def gettable(headrulesname):
    ptable = PropagationTable()
    if headrulesname == 'Ane_FTB4':
        ptable.add_mapping('AP', ['A','ADJWH'], 'R')
        ptable.add_mapping('AP', ['ET'], 'R')
        ptable.add_mapping('AP', ['VPP'], 'R')
        ptable.add_mapping('AP', ['ADV','ADVWH'], 'R')
        ptable.add_mapping('AdP', ['ADV','ADVWH'], 'R')
        ptable.add_mapping('NP', ['NC','NPP','PRO','PROWH','PROREL'], 'L')
        ptable.add_mapping('NP', ['NP'], 'L')
        ptable.add_mapping('NP', ['A','ADJWH'], 'L')
        ptable.add_mapping('NP', ['AP'], 'L')
        ptable.add_mapping('NP', ['I'], 'L')
        ptable.add_mapping('NP', ['VPpart'], 'L')
        ptable.add_mapping('NP', ['ADV','ADVWH'], 'L')
        ptable.add_mapping('NP', ['AdP'], 'L')
        ptable.add_mapping('NP', ['ET'], 'L')
        ptable.add_mapping('NP', ['DET','DETWH'], 'L')
        ptable.add_mapping('PP', ['P'], 'L')
        ptable.add_mapping('PP', ['P+D'], 'L')
        ptable.add_mapping('PP', ['NP','P+PRO'], 'L')
        ptable.add_mapping('Srel', ['VN'], 'R')
        ptable.add_mapping('Srel', ['AP'], 'R')
        ptable.add_mapping('Srel', ['NP'], 'R')
        ptable.add_mapping('Ssub', ['VN'], 'R')
        ptable.add_mapping('Ssub', ['AP'], 'R')
        ptable.add_mapping('Ssub', ['NP'], 'R')
        ptable.add_mapping('Ssub', ['PP'], 'R')
        ptable.add_mapping('Ssub', ['VPinf'], 'R')
        ptable.add_mapping('Ssub', ['Ssub'], 'R')
        ptable.add_mapping('Ssub', ['VPpart'], 'R')
        ptable.add_mapping('Ssub', ['A','ADJWH'], 'R')
        ptable.add_mapping('Ssub', ['ADV','ADVWH'], 'R')
        ptable.add_mapping('Sint', ['VN'], 'R')
        ptable.add_mapping('Sint', ['AP'], 'R')
        ptable.add_mapping('Sint', ['NP'], 'R')
        ptable.add_mapping('Sint', ['PP'], 'R')
        ptable.add_mapping('Sint', ['VPinf'], 'R')
        ptable.add_mapping('Sint', ['Ssub'], 'R')
        ptable.add_mapping('Sint', ['VPpart'], 'R')
        ptable.add_mapping('Sint', ['A','ADJWH'], 'R')
        ptable.add_mapping('Sint', ['ADV','ADVWH'], 'R')
        ptable.add_mapping('VPinf', ['VN'], 'L')
        ptable.add_mapping('VPinf', ['V','VIMP','VINF','VPP','VPR','VS'], 'L')
        ptable.add_mapping('VPpart', ['VPP','VPR'], 'L')
        ptable.add_mapping('VPpart', ['VN'], 'L')
        ptable.add_mapping('VN', ['V','VIMP','VINF','VPP','VPR','VS'], 'R')
        ptable.add_mapping('VN', ['VPinf'], 'R')
        ptable.add_mapping('COORD', ['CC','PONCT','CS'], 'L')
        ptable.add_mapping('SENT', ['VN'], 'R')
        ptable.add_mapping('SENT', ['AP'], 'R')
        ptable.add_mapping('SENT', ['NP'], 'R')
        ptable.add_mapping('SENT', ['Srel'], 'R')
        ptable.add_mapping('SENT', ['VPpart'], 'R')
        ptable.add_mapping('SENT', ['AdP'], 'R')
        ptable.add_mapping('SENT', ['I'], 'R')
        ptable.add_mapping('SENT', ['Ssub'], 'R')
        ptable.add_mapping('SENT', ['VPinf'], 'R')
        ptable.add_mapping('SENT', ['PP'], 'R')
        ptable.add_mapping('SENT', ['ADV','ADVWH'], 'R')
        ptable.add_mapping('PONCT', ['PONCT'], 'R')
    elif headrulesname == 'Ane_FTBARUN':
        ptable.add_mapping('AP', ['A'], 'R')
        ptable.add_mapping('AP', ['ET'], 'R')
        ptable.add_mapping('AP', ['V'], 'R')
        ptable.add_mapping('AP', ['ADV'], 'R')
        ptable.add_mapping('AdP', ['ADV'], 'R')
        ptable.add_mapping('NP', ['N','PRO'], 'L')
        ptable.add_mapping('NP', ['NP'], 'L')
        ptable.add_mapping('NP', ['A'], 'L')
        ptable.add_mapping('NP', ['AP'], 'L')
        ptable.add_mapping('NP', ['I'], 'L')
        ptable.add_mapping('NP', ['VPpart'], 'L')
        ptable.add_mapping('NP', ['ADV'], 'L')
        ptable.add_mapping('NP', ['AdP'], 'L')
        ptable.add_mapping('NP', ['ET'], 'L')
        ptable.add_mapping('NP', ['D'], 'L')
        ptable.add_mapping('PP', ['P'], 'L')
        ptable.add_mapping('PP', ['PC'], 'L')
        ptable.add_mapping('Srel', ['VN'], 'R')
        ptable.add_mapping('Srel', ['AP'], 'R')
        ptable.add_mapping('Srel', ['NP'], 'R')
        ptable.add_mapping('Ssub', ['VN'], 'R')
        ptable.add_mapping('Ssub', ['AP'], 'R')
        ptable.add_mapping('Ssub', ['NP'], 'R')
        ptable.add_mapping('Ssub', ['PP'], 'R')
        ptable.add_mapping('Ssub', ['VPinf'], 'R')
        ptable.add_mapping('Ssub', ['Ssub'], 'R')
        ptable.add_mapping('Ssub', ['VPpart'], 'R')
        ptable.add_mapping('Ssub', ['A'], 'R')
        ptable.add_mapping('Ssub', ['ADV'], 'R')
        ptable.add_mapping('Sint', ['VN'], 'R')
        ptable.add_mapping('Sint', ['AP'], 'R')
        ptable.add_mapping('Sint', ['NP'], 'R')
        ptable.add_mapping('Sint', ['PP'], 'R')
        ptable.add_mapping('Sint', ['VPinf'], 'R')
        ptable.add_mapping('Sint', ['Ssub'], 'R')
        ptable.add_mapping('Sint', ['VPpart'], 'R')
        ptable.add_mapping('Sint', ['A'], 'R')
        ptable.add_mapping('Sint', ['ADV'], 'R')
        ptable.add_mapping('VPinf', ['VN'], 'L')
        ptable.add_mapping('VPinf', ['V'], 'L')
        ptable.add_mapping('VPpart', ['V'], 'L')
        ptable.add_mapping('VPpart', ['VN'], 'L')
        ptable.add_mapping('VN', ['V'], 'R')
        ptable.add_mapping('VN', ['VPinf'], 'R')
        ptable.add_mapping('COORD', ['C','PONCT'], 'L')
        ptable.add_mapping('SENT', ['VN'], 'R')
        ptable.add_mapping('SENT', ['AP'], 'R')
        ptable.add_mapping('SENT', ['NP'], 'R')
        ptable.add_mapping('SENT', ['Srel'], 'R')
        ptable.add_mapping('SENT', ['VPpart'], 'R')
        ptable.add_mapping('SENT', ['AdP'], 'R')
        ptable.add_mapping('SENT', ['I'], 'R')
        ptable.add_mapping('SENT', ['Ssub'], 'R')
        ptable.add_mapping('SENT', ['VPinf'], 'R')
        ptable.add_mapping('SENT', ['PP'], 'R')
        ptable.add_mapping('SENT', ['ADV'], 'R')
        ptable.add_mapping('PONCT', ['*'], 'L')
    elif headrulesname == 'Ane_MFT4':
        ptable.add_mapping('AP', ['A','ADJWH'], 'R')
        ptable.add_mapping('AP', ['ET'], 'R')
        ptable.add_mapping('AP', ['VPP'], 'R')
        ptable.add_mapping('AP', ['ADV','ADVWH'], 'R')
        ptable.add_mapping('AP', ['PREF'], 'L')
        ptable.add_mapping('AP', ['COORD'], 'L')
        ptable.add_mapping('AdP', ['ADV','ADVWH'], 'R')
        ptable.add_mapping('AdP', ['COORD'], 'L')
        ptable.add_mapping('NP', ['NCC','NPP','PRO','PROWH','PROREL'], 'L')
        ptable.add_mapping('NP', ['NP'], 'L')
        ptable.add_mapping('NP', ['A','ADJWH'], 'L')
        ptable.add_mapping('NP', ['AP'], 'L')
        ptable.add_mapping('NP', ['I'], 'L')
        ptable.add_mapping('NP', ['VPpart'], 'L')
        ptable.add_mapping('NP', ['ADV','ADVWH'], 'L')
        ptable.add_mapping('NP', ['AdP'], 'L')
        ptable.add_mapping('NP', ['ET'], 'L')
        ptable.add_mapping('NP', ['DET','DETWH'], 'L')
        ptable.add_mapping('NP', ['COORD'], 'L')
        ptable.add_mapping('PP', ['P'], 'L')
        ptable.add_mapping('PP', ['P+D'], 'L')
        ptable.add_mapping('PP', ['NP','P+PRO'], 'L')
        ptable.add_mapping('PP', ['COORD'], 'L')
        ptable.add_mapping('Srel', ['VN'], 'R')
        ptable.add_mapping('Srel', ['AP'], 'R')
        ptable.add_mapping('Srel', ['NP'], 'R')
        ptable.add_mapping('Srel', ['COORD'], 'L')
        ptable.add_mapping('Ssub', ['VN'], 'R')
        ptable.add_mapping('Ssub', ['AP'], 'R')
        ptable.add_mapping('Ssub', ['NP'], 'R')
        ptable.add_mapping('Ssub', ['PP'], 'R')
        ptable.add_mapping('Ssub', ['VPinf'], 'R')
        ptable.add_mapping('Ssub', ['Ssub'], 'R')
        ptable.add_mapping('Ssub', ['VPpart'], 'R')
        ptable.add_mapping('Ssub', ['A','ADJWH'], 'R')
        ptable.add_mapping('Ssub', ['ADV','ADVWH'], 'R')
        ptable.add_mapping('Ssub', ['COORD'], 'L')
        ptable.add_mapping('Sint', ['VN'], 'R')
        ptable.add_mapping('Sint', ['AP'], 'R')
        ptable.add_mapping('Sint', ['NP'], 'R')
        ptable.add_mapping('Sint', ['PP'], 'R')
        ptable.add_mapping('Sint', ['VPinf'], 'R')
        ptable.add_mapping('Sint', ['Ssub'], 'R')
        ptable.add_mapping('Sint', ['VPpart'], 'R')
        ptable.add_mapping('Sint', ['A','ADJWH'], 'R')
        ptable.add_mapping('Sint', ['ADV','ADVWH'], 'R')
        ptable.add_mapping('Sint', ['COORD'], 'L')
        ptable.add_mapping('VPinf', ['VN'], 'L')
        ptable.add_mapping('VPinf', ['V','VIMP','VINF','VPP','VPR','VS'], 'L')
        ptable.add_mapping('VPinf', ['COORD'], 'L')
        ptable.add_mapping('VPpart', ['VPP','VPR'], 'L')
        ptable.add_mapping('VPpart', ['VN'], 'L')
        ptable.add_mapping('VPpart', ['COORD'], 'L')
        ptable.add_mapping('VN', ['V','VIMP','VINF','VPP','VPR','VS'], 'R')
        ptable.add_mapping('VN', ['VPinf'], 'R')
        ptable.add_mapping('COORD', ['CC','PONCT','CS'], 'L')
        ptable.add_mapping('SENT', ['VN'], 'R')
        ptable.add_mapping('SENT', ['AP'], 'R')
        ptable.add_mapping('SENT', ['NP'], 'R')
        ptable.add_mapping('SENT', ['Srel'], 'R')
        ptable.add_mapping('SENT', ['VPpart'], 'R')
        ptable.add_mapping('SENT', ['AdP'], 'R')
        ptable.add_mapping('SENT', ['I'], 'R')
        ptable.add_mapping('SENT', ['Ssub'], 'R')
        ptable.add_mapping('SENT', ['VPinf'], 'R')
        ptable.add_mapping('SENT', ['PP'], 'R')
        ptable.add_mapping('SENT', ['ADV','ADVWH'], 'R')
        ptable.add_mapping('PONCT', ['PONCT'], 'R')
        ptable.add_mapping('NC', ['*'], 'L')
        ptable.add_mapping('VP', ['VN'], 'L')
    elif headrulesname == 'Ane_MFTMIN':
        ptable.add_mapping('AP', ['A'], 'R')
        ptable.add_mapping('AP', ['ET'], 'R')
        ptable.add_mapping('AP', ['VPpart','V'], 'R')
        ptable.add_mapping('AP', ['ADV'], 'R')
        ptable.add_mapping('AP', ['COORD'], 'L')
        ptable.add_mapping('AP', ['PREF'], 'L')
        ptable.add_mapping('AdP', ['ADV'], 'R')
        ptable.add_mapping('AdP', ['COORD'], 'L')
        ptable.add_mapping('NP', ['N','PRO'], 'L')
        ptable.add_mapping('NP', ['NP'], 'L')
        ptable.add_mapping('NP', ['A'], 'L')
        ptable.add_mapping('NP', ['AP'], 'L')
        ptable.add_mapping('NP', ['I'], 'L')
        ptable.add_mapping('NP', ['VPpart'], 'L')
        ptable.add_mapping('NP', ['ADV'], 'L')
        ptable.add_mapping('NP', ['AdP'], 'L')
        ptable.add_mapping('NP', ['N'], 'L')
        ptable.add_mapping('NP', ['ET'], 'L')
        ptable.add_mapping('NP', ['D'], 'L')
        ptable.add_mapping('NP', ['COORD'], 'L')
        ptable.add_mapping('PP', ['P'], 'L')
        ptable.add_mapping('PP', ['P+D'], 'L')
        ptable.add_mapping('PP', ['NP','P+PRO'], 'L')
        ptable.add_mapping('PP', ['COORD'], 'L')
        ptable.add_mapping('Srel', ['VN'], 'R')
        ptable.add_mapping('Srel', ['AP'], 'R')
        ptable.add_mapping('Srel', ['NP'], 'R')
        ptable.add_mapping('Srel', ['VN'], 'R')
        ptable.add_mapping('Srel', ['COORD'], 'L')
        ptable.add_mapping('Ssub', ['VN'], 'R')
        ptable.add_mapping('Ssub', ['AP'], 'R')
        ptable.add_mapping('Ssub', ['NP'], 'R')
        ptable.add_mapping('Ssub', ['PP'], 'R')
        ptable.add_mapping('Ssub', ['VPinf'], 'R')
        ptable.add_mapping('Ssub', ['Ssub'], 'R')
        ptable.add_mapping('Ssub', ['VPpart'], 'R')
        ptable.add_mapping('Ssub', ['A'], 'R')
        ptable.add_mapping('Ssub', ['ADV'], 'R')
        ptable.add_mapping('Ssub', ['VN'], 'R')
        ptable.add_mapping('Sint', ['VN'], 'R')
        ptable.add_mapping('Sint', ['AP'], 'R')
        ptable.add_mapping('Sint', ['NP'], 'R')
        ptable.add_mapping('Sint', ['PP'], 'R')
        ptable.add_mapping('Sint', ['VPinf'], 'R')
        ptable.add_mapping('Sint', ['Ssub'], 'R')
        ptable.add_mapping('Sint', ['VPpart'], 'R')
        ptable.add_mapping('Sint', ['A'], 'R')
        ptable.add_mapping('Sint', ['ADV'], 'R')
        ptable.add_mapping('Sint', ['VN'], 'R')
        ptable.add_mapping('VPinf', ['VN'], 'L')
        ptable.add_mapping('VPinf', ['VPinf','V'], 'L')
        ptable.add_mapping('VPinf', ['VPpart','V'], 'L')
        ptable.add_mapping('VPinf', ['VN'], 'R')
        ptable.add_mapping('VPpart', ['VPpart','V'], 'L')
        ptable.add_mapping('VPpart', ['VN'], 'L')
        ptable.add_mapping('VN', ['VPpart','VPinf','V','VN'], 'R')
        ptable.add_mapping('VN', ['VPinf'], 'R')
        ptable.add_mapping('COORD', ['C','PONCT'], 'L')
        ptable.add_mapping('SENT', ['VN'], 'R')
        ptable.add_mapping('SENT', ['AP'], 'R')
        ptable.add_mapping('SENT', ['NP'], 'R')
        ptable.add_mapping('SENT', ['Srel'], 'R')
        ptable.add_mapping('SENT', ['VPpart'], 'R')
        ptable.add_mapping('SENT', ['AdP'], 'R')
        ptable.add_mapping('SENT', ['I'], 'R')
        ptable.add_mapping('SENT', ['Ssub'], 'R')
        ptable.add_mapping('SENT', ['VPinf'], 'R')
        ptable.add_mapping('SENT', ['PP'], 'R')
        ptable.add_mapping('SENT', ['ADV'], 'R')
        ptable.add_mapping('SENT', ['VN'], 'R')
        ptable.add_mapping('PONCT', ['*'], 'L')
        ptable.add_mapping('NC', ['*'], 'L')
        ptable.add_mapping('VP', ['VN'], 'L')
    elif headrulesname == 'Ane_MFTSCHLU':
	# Marie ajout 
        #ptable.add_mapping('AP', ['A','A_int'], 'R')
        ptable.add_mapping('AP', ['A','A_int','A_card'], 'R')
        ptable.add_mapping('AP', ['ET'], 'R')
        ptable.add_mapping('AP', ['VPpart','V_part'], 'R')
        ptable.add_mapping('AP', ['ADV','ADV_int'], 'R')
        ptable.add_mapping('AP', ['COORD','COORD_AP','COORD_unary','COORD_NP','COORD_PP','COORD_VPpart','COORD_VPinf','COORD_Srel','COORD_Sint','COORD_Ssub','COORD_VP','COORD_AdP','COORD_UC','COORD_NC'], 'L')
        ptable.add_mapping('AP', ['PREF'], 'L')
        ptable.add_mapping('AP_int', ['A','A_int'], 'R')
        ptable.add_mapping('AP_int', ['ET'], 'R')
        ptable.add_mapping('AP_int', ['VPpart','V_part'], 'R')
        ptable.add_mapping('AP_int', ['ADV','ADV_int'], 'R')
        ptable.add_mapping('AP_int', ['COORD','COORD_AP','COORD_unary','COORD_NP','COORD_PP','COORD_VPpart','COORD_VPinf','COORD_Srel','COORD_Sint','COORD_Ssub','COORD_VP','COORD_AdP','COORD_UC','COORD_NC'], 'L')
        ptable.add_mapping('AP_int', ['PREF'], 'L')
        ptable.add_mapping('AdP', ['ADV','ADV_int'], 'R')
        ptable.add_mapping('AdP', ['COORD','COORD_AdP','COORD_unary','COORD_NP','COORD_AP','COORD_PP','COORD_VPpart','COORD_VPinf','COORD_Srel','COORD_Sint','COORD_Ssub','COORD_VP','COORD_UC','COORD_NC'], 'L')
        ptable.add_mapping('AdP_int', ['ADV','ADV_int'], 'R')
        ptable.add_mapping('AdP_int', ['COORD','COORD_AdP','COORD_unary','COORD_NP','COORD_AP','COORD_PP','COORD_VPpart','COORD_VPinf','COORD_Srel','COORD_Sint','COORD_Ssub','COORD_VP','COORD_UC','COORD_NC'], 'L')
        ptable.add_mapping('NP', ['N','N_card','PRO','PRO_rel','PRO_int','PRO_card'], 'L')
        ptable.add_mapping('NP', ['CL'], 'L')
        # MARIE : ajout post Ane_MFTSCHLU
        ptable.add_mapping('NP', ['NP'], 'L')
        ptable.add_mapping('NP', ['A'], 'L')
        ptable.add_mapping('NP', ['AP'], 'L')
        ptable.add_mapping('NP', ['I'], 'L')
        ptable.add_mapping('NP', ['VPpart'], 'L')
        ptable.add_mapping('NP', ['ADV'], 'L')
        ptable.add_mapping('NP', ['AdP'], 'L')
        ptable.add_mapping('NP', ['N'], 'L')
        ptable.add_mapping('NP', ['ET'], 'L')
        ptable.add_mapping('NP', ['D'], 'L')
        ptable.add_mapping('NP', ['COORD','COORD_unary','COORD_NP','COORD_AP','COORD_PP','COORD_VPpart','COORD_VPinf','COORD_Srel','COORD_Sint','COORD_Ssub','COORD_VP','COORD_AdP','COORD_UC','COORD_NC'], 'L')
        ptable.add_mapping('NP_rel', ['N','N_card','PRO','PRO_rel','P+PRO_rel','PRO_int','PRO_card'], 'L')
        ptable.add_mapping('NP_rel', ['NP'], 'L')
        ptable.add_mapping('NP_rel', ['A'], 'L')
        ptable.add_mapping('NP_rel', ['AP'], 'L')
        ptable.add_mapping('NP_rel', ['I'], 'L')
        ptable.add_mapping('NP_rel', ['VPpart'], 'L')
        ptable.add_mapping('NP_rel', ['ADV'], 'L')
        ptable.add_mapping('NP_rel', ['AdP'], 'L')
        ptable.add_mapping('NP_rel', ['N'], 'L')
        ptable.add_mapping('NP_rel', ['ET'], 'L')
        ptable.add_mapping('NP_rel', ['D'], 'L')
        ptable.add_mapping('NP_rel', ['COORD','COORD_unary','COORD_NP','COORD_AP','COORD_PP','COORD_VPpart','COORD_VPinf','COORD_Srel','COORD_Sint','COORD_Ssub','COORD_VP','COORD_AdP','COORD_UC','COORD_NC'], 'L')
        ptable.add_mapping('NP_int', ['N','N_card','PRO','PRO_rel','PRO_int','PRO_card'], 'L')
        ptable.add_mapping('NP_int', ['NP'], 'L')
        ptable.add_mapping('NP_int', ['A'], 'L')
        ptable.add_mapping('NP_int', ['AP'], 'L')
        ptable.add_mapping('NP_int', ['I'], 'L')
        ptable.add_mapping('NP_int', ['VPpart'], 'L')
        ptable.add_mapping('NP_int', ['ADV'], 'L')
        ptable.add_mapping('NP_int', ['AdP'], 'L')
        ptable.add_mapping('NP_int', ['N'], 'L')
        ptable.add_mapping('NP_int', ['ET'], 'L')
        ptable.add_mapping('NP_int', ['D'], 'L')
        ptable.add_mapping('NP_int', ['COORD','COORD_unary','COORD_NP','COORD_AP','COORD_PP','COORD_VPpart','COORD_VPinf','COORD_Srel','COORD_Sint','COORD_Ssub','COORD_VP','COORD_AdP','COORD_UC','COORD_NC'], 'L')
        ptable.add_mapping('PP', ['P'], 'L')
        ptable.add_mapping('PP', ['P+D'], 'L')
        ptable.add_mapping('PP', ['NP','P+PRO'], 'L')
        ptable.add_mapping('PP', ['COORD_unary','COORD_NP','COORD_AP','COORD_PP','COORD_VPpart','COORD_VPinf','COORD_Srel','COORD_Sint','COORD_Ssub','COORD_VP','COORD_AdP','COORD_UC','COORD_NC'], 'L')
        ptable.add_mapping('PP_rel', ['P'], 'L')
        ptable.add_mapping('PP_rel', ['P+D'], 'L')
        # ajout marie
        #ptable.add_mapping('PP_rel', ['NP_rel','P+PRO_rel'], 'L')
        ptable.add_mapping('PP_rel', ['NP_rel','PRO_rel','P+PRO_rel'], 'L')
        ptable.add_mapping('PP_rel', ['COORD','COORD_unary','COORD_NP','COORD_AP','COORD_PP','COORD_VPpart','COORD_VPinf','COORD_Srel','COORD_Sint','COORD_Ssub','COORD_VP','COORD_AdP','COORD_UC','COORD_NC'], 'L')
        ptable.add_mapping('PP_int', ['P'], 'L')
        ptable.add_mapping('PP_int', ['P+D'], 'L')
        ptable.add_mapping('PP_int', ['NP_int','P+PRO_int'], 'L')
        ptable.add_mapping('PP_int', ['COORD','COORD_unary','COORD_NP','COORD_AP','COORD_PP','COORD_VPpart','COORD_VPinf','COORD_Srel','COORD_Sint','COORD_Ssub','COORD_VP','COORD_AdP','COORD_UC','COORD_NC'], 'L')
        ptable.add_mapping('Srel', ['VN_finite','COORD_VN_finite'], 'R')
        ptable.add_mapping('Srel', ['AP'], 'R')
        ptable.add_mapping('Srel', ['NP','NP_rel'], 'R')
        ptable.add_mapping('Srel', ['VN'], 'R')
        ptable.add_mapping('Srel', ['COORD','COORD_unary','COORD_NP','COORD_AP','COORD_PP','COORD_VPpart','COORD_VPinf','COORD_Srel','COORD_Sint','COORD_Ssub','COORD_VP','COORD_AdP','COORD_UC','COORD_NC'], 'L')
        ptable.add_mapping('Ssub', ['VN_finite','COORD_VN_finite'], 'R')
        # ajout marie
        ptable.add_mapping('Ssub', ['COORD_Ssub'], 'R')
        ptable.add_mapping('Ssub', ['AP'], 'R')
        ptable.add_mapping('Ssub', ['NP'], 'R')
        ptable.add_mapping('Ssub', ['PP'], 'R')
        ptable.add_mapping('Ssub', ['VPinf'], 'R')
        ptable.add_mapping('Ssub', ['Ssub'], 'R')
        ptable.add_mapping('Ssub', ['VPpart'], 'R')
        ptable.add_mapping('Ssub', ['A'], 'R')
        ptable.add_mapping('Ssub', ['ADV'], 'R')
        ptable.add_mapping('Ssub', ['VN'], 'R')
        ptable.add_mapping('Sint', ['VN_finite','COORD_VN_finite'], 'R')
        ptable.add_mapping('Sint', ['AP'], 'R')
        ptable.add_mapping('Sint', ['NP'], 'R')
        ptable.add_mapping('Sint', ['PP'], 'R')
        ptable.add_mapping('Sint', ['VPinf'], 'R')
        ptable.add_mapping('Sint', ['Ssub'], 'R')
        ptable.add_mapping('Sint', ['VPpart'], 'R')
        ptable.add_mapping('Sint', ['A'], 'R')
        ptable.add_mapping('Sint', ['ADV'], 'R')
        ptable.add_mapping('Sint', ['VN'], 'R')
        ptable.add_mapping('VPinf', ['VN_inf','VN'], 'L')
        ptable.add_mapping('VPinf', ['VPinf','V_inf'], 'L')
        ptable.add_mapping('VPinf', ['VPpart','V','COORD_VPinf','COORD_VN_inf','COORD'], 'L')
        ptable.add_mapping('VPpart', ['VPpart','V_part','V'], 'L')
        ptable.add_mapping('VPpart', ['VN_part','VN', 'COORD_VPpart','COORD_VN_part','COORD'], 'L')
        # pou nopropag...
        ptable.add_mapping('VN', ['V_finite','COORD','V_inf','V_part'], 'R')
        ptable.add_mapping('VN_finite', ['V_finite','VN_finite','COORD_VN_finite','COORD'], 'R')
        ptable.add_mapping('VN_inf', ['V_inf','VN_inf','COORD_VN_inf'], 'R')
        ptable.add_mapping('VN_part', ['V_part','VN_part','COORD_VN_part','COORD'], 'R')
        # ajout marie
        ptable.add_mapping('COORD', ['C_C','PONCT'], 'L')
        ptable.add_mapping('COORD_unary', ['C_C','PONCT'], 'L')
        # ajout marie 
        #ptable.add_mapping('COORD_NP', ['C_C','PONCT'], 'L')
        ptable.add_mapping('COORD_NP', ['C_C','PONCT','NP'], 'L')
        ptable.add_mapping('COORD_PP', ['C_C','PONCT'], 'L')
        ptable.add_mapping('COORD_AP', ['C_C','PONCT'], 'L')
        ptable.add_mapping('COORD_AdP', ['C_C','PONCT'], 'L')
        ptable.add_mapping('COORD_VPpart', ['C_C','PONCT'], 'L')
        ptable.add_mapping('COORD_VPinf', ['C_C','PONCT'], 'L')
        ptable.add_mapping('COORD_VP', ['C_C','PONCT'], 'L')
        ptable.add_mapping('COORD_Ssub', ['C_C','PONCT'], 'L')
        ptable.add_mapping('COORD_Srel', ['C_C','PONCT'], 'L')
        ptable.add_mapping('COORD_Sint', ['C_C','PONCT'], 'L')
        ptable.add_mapping('COORD_UC', ['C_C','PONCT'], 'L')
        ptable.add_mapping('COORD_NC', ['C_C','PONCT'], 'L')
        ptable.add_mapping('SENT', ['VN_finite','VN_inf','VN_part'], 'R')
        ptable.add_mapping('SENT', ['AP'], 'R')
        ptable.add_mapping('SENT', ['NP'], 'R')
        ptable.add_mapping('SENT', ['Srel'], 'R')
        ptable.add_mapping('SENT', ['VPpart'], 'R')
        ptable.add_mapping('SENT', ['AdP'], 'R')
        ptable.add_mapping('SENT', ['I'], 'R')
        ptable.add_mapping('SENT', ['Ssub'], 'R')
        ptable.add_mapping('SENT', ['VPinf'], 'R')
        ptable.add_mapping('SENT', ['PP'], 'R')
        ptable.add_mapping('SENT', ['ADV'], 'R')
        ptable.add_mapping('SENT', ['VN'], 'R')
        # ajout marie
        ptable.add_mapping('SENT', ['COORD_Sint','COORD_Ssub','COORD_unary','COORD_UC','COORD_NP','COORD_AP','COORD'], 'R')
        ptable.add_mapping('PONCT', ['*'], 'L')
        ptable.add_mapping('NC', ['*'], 'L')
        ptable.add_mapping('VP', ['VN_finite'], 'L')
        return ptable

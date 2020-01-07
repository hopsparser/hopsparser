#!/usr/bin/env python3

from PropagTable import *
import sys
from constree import *



#INPUT
def mcvf_readtree(istream):
    """
    Reads the next tree (or return false if eofstream is reached)
    """
    tree_str = ''
    brackets = 0
    bfr = istream.readline()
    while bfr:
        bfr = bfr.split('//')[0]
        if not (bfr=='' or bfr.startswith("***") or bfr.isspace()):
            tree_str += bfr
            brackets += bfr.count('(')
            brackets -= bfr.count(')')
            if brackets == 0:
                tree_str = ' '.join(tree_str.split())
                tree_str = '(ROOT '+tree_str[1:]
        return ConsTree.read_tree(tree_str) #<-----
        bfr = istream.readline()
    return False

def mcvf_read(istream):
    """
    Reads an MCVF treebank and returns a tree list
    """
    tlist = [] 
    tree = mcvf_readtree(istream)
    while tree:
        tlist.append(tree)
        tree = mcvf_readtree(istream)
    return tlist

def mcvf_heads():
    """
    head annotation table for MCVF
    """
    ptable = PropagationTable()

    #################################################
    #   SACHA SECTION (BEGIN)
    #################################################
    
    ptable.add_mapping("ADJP",["ADJ", "*ICH*", "VPP", "VG", "ADJNUM", "ADJP", "ADJS", "ADJR", "ADJZ", "ADJX", "PP", "QP", "QR", "CONJP"],"R")
    ptable.add_mapping("ADVP",["ADV", "ADVR", "ADVNEG", "WADV", "WADVP", "QR", "*ICH*", "*T*", "P", "FP", "PRO", "PP", "CP-FRL", "CP-REL", "IP-MAT", "CONJP"],"L")
    ptable.add_mapping("CONJP",["CONJO", "IP", "NP", "ADVP", "*ICH*", "CP", "ADJP", "VX", "XX", "PP"], "L")
    ptable.add_mapping("CP",["CONJS", "WNP", "WPP", "*ICH", "ADVP", "WADVP", "IP", "CP", "*T*", "WADJP", "WQP", "CONJP"], "L")
    ptable.add_mapping("CMPP",["CMP", "CONJP"], "L") 
    ptable.add_mapping("IP",["VJ", "AJ", "MDJ", "VX", "LJ", "EJ", "IP", "VPP", "VG", "PPL", "NP-PRD", "ADJ-PRD", "ADJP-PRD", "CONJP"], "L") 
    ptable.add_mapping("NP",["DAT", "NCS", "NCPL", "NPRS", "NPRPL", "PRO", "CL-NP-ACC", "CL-NP-SBJ", "CL-NP-DTV", "CL-NP-RFL", "CL-PP", "*pro*", "*ICH*", "*con*", "*T*", "*proimp*", "Q", "CP-FRL", "VPP", "QR", "*arb*", "ADJNUM", "QP", "NUM", "ADJZ", "PROIMP", "ADJR", "D", "QTP", "ADJ", "ETR", "*", "DZ", "NP", "IP", "CP", "VG", "DF", "WPP", "CONJP"], "L")  
    ptable.add_mapping("ITJP",["ITJ", "NCS", "ADV", "PP", "WADV", "NPRS", "NP", "VP", "ITJP", "*ICH*", "VJ"],"L")
    ptable.add_mapping("PP",["P", "*ICH*", "*T*", "PR", "PRO", "CMP", "DF", "ETR",  "PP", "CONJP"], "L")  
    ptable.add_mapping("QP",["Q", "QR", "*ICH*", "*T*", "CONJP"], "L")  
    ptable.add_mapping("WADJP",["WADV", "*T*", "ADV"], "L")
    ptable.add_mapping("WADVP",["ADV", "WADV", "WPRO", "0", "*T*", "WADVP", "WPP"], "R")  
    ptable.add_mapping("WNP",["WPRO", "NCS", "NCPL", "WNP", "NP", "CONJP"], "R")  
    ptable.add_mapping("WPP",["P", "WPRO", "*T*", "CONJP"], "L")  
    ptable.add_mapping("WQP",["Q", "WPRO", "*T*", "CONJP"], "L")
    #################################################
    #   SACHA SECTION (END)
    #################################################
    return ptable

def head_annotate(tree,ptable,debug=True,show=False,flow=None):
    """
    Performs head annotation : modifies (side effect, destructive)
    the trees by adding a 'head' attribute to tree nodes 
    """
    if not tree.is_leaf():
        idx = ptable.head_index(tree,tree.children)  
        for child in tree.children:
            child.is_head = False
            head_annotate(child,ptable,debug,False)
        tree.children[idx].is_head = True
        if debug and not tree.children[idx].is_leaf():
            tree.children[idx].label+="[H]"
    if show:
        flow.write(str(tree))

                
def head_annotate_all(tree_list, debug=True,show=False,flow=None):
    """
    Performs head annotation : modifies (side effect, destructive)
    the trees by adding a 'head' attribute to tree nodes 
    """
    table = mcvf_heads()
    for tree in tree_list:
        constree=tree[0]
        head_annotate(constree,table,debug,show,flow)
        constree.is_head=True
        if debug:
            constree.label+='[H]'
        print(constree)

                
def head_annotate_all2(tree_list, debug=True,show=False,flow=None):
    """
    Performs head annotation : modifies (side effect, destructive)
    the trees by adding a 'head' attribute to tree nodes 
    """
    table = mcvf_heads()
    for tree in tree_list:
        head_annotate(tree,table,debug,show,flow)
        tree.is_head=True
        if debug:
            tree.label+='[H]'

        
def pretty_print_all(treelist,ostream=sys.stdout):
    """
    prints all trees
    """
    for tree in treelist:
        ostream.write(str(tree)+'\n')




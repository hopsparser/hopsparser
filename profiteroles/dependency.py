from mcvf_heads import *
from constree import *
from PropagTable import *

class deptree:
	def __init__(self):
		self.root=None
		self.nodes=[]

class Node:
	def __init__(self):
		self.deps=[]
		self.gov=None
		self.label=("","")

def extract_dependencies(constree,label=False):
	'''
	returns a list of dependenceies from a constree:
	a dependency: a tuple (index of child, index of governor)
	'''
	dependencies=[]
	labels=[]
	if constree.is_leaf():
		return[]
	for children in constree.children:
		if not children.is_head:
			if label:
				dependencies.append((children.idx,children.label))
				labels.append((children.idx,constree.label+"/"+constree.label))
			else:		
				dependencies.append((children.idx,constree.idx))
			
		dependencies.extend(extract_dependencies(children,label))

	if label:
		return dependencies,labels
	return dependencies

def add_root(depList):
	'''
	adds the dependency 
	roots governs itself
	dep=(root,  root)
	'''
	root=0
	for i in range(len(depList)): 
	 	if depList[i][1]!=i:
	 		depList.insert(i,(0,i))
	 		root=i
	 		break

def index_leaves(toklist):
    idx = 0
    for word in toklist:
        #print('   ',word)
        if word.label[0] != '*' and word.label[-1] != '*':
            word.idx = idx
            idx += 1
        else:
            word.idx = -1
            
def index_tree(tree):

    if not tree.is_leaf():
        tree.idx = -1
    for child in tree.children:
        index_tree(child)
        if child.is_head:
            tree.idx = child.idx
            
def strip_labels(tree):  
    """
    Strips the labels and returns a dictionary mapping trace indexes
    to their nodes
    """
    fillers = {} #maps idx to tree nodes
    tree.longlabel = tree.label
    if tree.label != '-':
        trace_idx      = tree.label.split('-')[-1]
        tree.label     = tree.label.split("-")[0] 
        #print('    ',tree.label,trace_idx)
        if tree.children and trace_idx in "0123456789":
            if tree.label[0] != '*' and tree.label[-1] != '*':
                fillers[trace_idx] = tree
            elif tree.label[0] == '*' and tree.label[-1] == '*':
                tree.filleridx = trace_idx
    for child in tree.children:
        fillers.update(strip_labels(child))
    return fillers

def extract_deps(tree,fillers):
    
    deps = [ ]     
    for child in tree.children:
        if child.idx != tree.idx:
            if child.idx == -1 and hasattr(child,'filleridx'): # gap !
                deps.append((fillers[child.filleridx].idx,tree.idx)) # fills the gap
            elif child.idx != -1:
                deps.append((child.idx,tree.idx)) # dep -> gov
        deps.extend( extract_deps(child,fillers) )
    return deps 

def generate_conll(ostream,tokens,child2gov):
    #Generates the CONLL-U#
    #print(child2gov)
    for elt in tokens:
        idx = elt.idx
        if idx != -1:
            root_idx = str(int(child2gov[idx])+1) if idx in child2gov else '0'
            line = [str(int(idx)+1),elt.label,'-','-','-','-',root_idx,'-','-','-'] 
            print('\t'.join(line),file=ostream)
    print('',file=ostream)
        
def main():
    '''
    runs the conll extraction of all trees from the mcvf treebank
    '''
    table = mcvf_heads()
    decoratedTrees=open("headAnnotated.mrg",'w')
    #stockage des arbres non annotés depuis le fichier .mrg
    mergedfile=open("mcvf.mrg","r")

	#3 tests tree for debugs:
    #testTrees=(mcvf_read(merged))
	#annotation du constree, sans modification des étiquettes
    #head_annotate_all(testTrees,debug=True)
    ostream = open("mcvf.conll","w")
	#écriture du conll
    for line in mergedfile:
        try:
            constree = ConsTree.read_tree(line)[0]
            #print(constree)
            fillers = strip_labels(constree)
            #print(constree)
            #print(fillers)
            head_annotate(constree,table,debug=False)
            #print(constree)
            tokens = constree.tokens(labels=False)
            index_leaves(tokens)
            index_tree(constree)
            child2gov = dict( extract_deps(constree,fillers) )
            generate_conll(ostream,tokens,child2gov)
            #print()
        except IndexError:
            print('Invalid tree (skipped)')
            
    ostream.close()
    mergedfile.close()
    
main()

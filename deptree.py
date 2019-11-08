

class DepGraph:
    def __init__(self,edges,wordlist=None,pos_tags=None,with_root=False):
        
        self.gov2dep = { }
        self.has_gov = set()            #set of nodes with a governor

        for (gov,label,dep) in edges:
            self.add_arc(gov,label,dep)

        if with_root:
            self.add_root()

        self.words    = wordlist
        self.pos_tags = pos_tags
        
    def fastcopy(self):
        """
        copy edges only not word nor tags
        """
        edgelist = list(self.gov2dep.values())
        flatlist = [edge for sublist in edgelist for edge in sublist]
        return DepGraph(flatlist)
        
    def get_all_labels(self):
        """
        Returns the list of dependency labels found on the arcs
        """
        all_labels = [ ] 
        for gov in self.gov2dep:
            all_labels.extend( [label for (gov,label,dep) in self.gov2dep[gov] ] )
        return all_labels
        
    def get_arc(self,gov,dep):
        """
        Returns the arc between gov and dep if it exists or None otherwise
        Args:
            gov (int): node idx
            dep (int): node idx
        Returns:
            A triple (gov,label,dep) or None.
        """
        if gov in self.gov2dep:
            for (_gov,deplabel,_dep) in self.gov2dep[gov]:
                if _dep == dep:
                    return (_gov,deplabel,_dep)
        return None

    def add_root(self):
            
        if -1 not in self.gov2dep:
            root = list(set(self.gov2dep) - self.has_gov)
            if len(root) == 1:
                self.add_arc(-1,'root',root[0])
            elif len(root) == 0:
                self.add_arc(-1,'root',0)
            else:
                print(len(root)) 
                assert(False) #no single root... problem.
            
    def add_arc(self,gov,label,dep):
        """
        Adds an arc to the dep graph
        """
        if gov in self.gov2dep:
            self.gov2dep[gov].append( (gov,label,dep) )
        else:
            self.gov2dep[gov] = [(gov,label,dep)]
            
        self.has_gov.add(dep)
        
    def is_cyclic_add(self,gov,dep):
        """
        Checks if the addition of an arc from gov to dep would create
        a cycle in the dep tree
        """
        return gov in self.span(dep)

    def is_dag_add(self,gov,dep):
        """
        Checks if the addition of an arc from gov to dep would create
        a Dag
        """
        return dep in self.has_gov

    def span(self,gov):
        """
        Returns the list of nodes in the yield of this node
        the set of j such that (i -*> j). 
        """
        agenda = [gov]
        closure = set( [gov] )
        while agenda:
           node = agenda.pop( )
           succ = [ dep  for (gov,label,dep) in self.gov2dep[node] ] if node in self.gov2dep else [ ]
           agenda.extend( [node for node in succ if node not in closure])
           closure.update(succ)
        return closure

    def _gap_degree(self,node):
        """
        Returns the gap degree of a node
        Args :
            node (int): a dep tree node
        """
        nspan = list(self.span(node))
        nspan.sort()
        gd = 0
        for idx in range(len(nspan)):
            if idx > 0:
              if nspan[idx]-nspan[idx-1] > 1:
                  gd += 1
        return gd
                  
    def gap_degree(self):
        """
        Returns the gap degree of a tree (suboptimal)
        """
        return max(self._gap_degree(node) for node in self.gov2dep)

    def is_projective(self):
        """
        Returns true if this tree is projective
        """
        return self.gap_degree() == 0
    
    @staticmethod
    def read_tree(istream):
        """
        Reads a conll tree from input stream 
        """
        conll = [ ]
        line  = istream.readline( )
        while istream and line.isspace( ):
            line  = istream.readline()
        while istream and not line.strip() == '':
            conll.append( line.split('\t'))
            line  = istream.readline()
        if not conll:
            return None
        words   = [ ]
        postags = [ ]
        edges   = [ ]
        for dataline in conll:
            words.append(dataline[1])
            if dataline[3] != '-':
                postags.append(dataline[3])
            if dataline[6] != '0': #do not add root immediately
                edges.append((int(dataline[6])-1,dataline[7],int(dataline[0])-1))
        return DepGraph(edges,words,pos_tags=postags,with_root=True)
    
    def __str__(self):
        """
        Conll string for the dep tree
        """
        lines    = [ ]
        revdeps  = dict([( dep, (label,gov) ) for node in self.gov2dep for (gov,label,dep) in self.gov2dep[node] ])
        for node in range( len(self.words)  ):
            L    = ['-']*11
            L[0] = str(node+1)
            L[1] = self.words[node]
            if self.pos_tags:
                L[3] = self.pos_tags[node]
            label,head = revdeps[node] if node in revdeps else ('root', -1)
            L[6] = str(head+1)
            L[7] = label
            lines.append( '\t'.join(L) ) 
        return '\n'.join(lines)

    def __len__(self):
        return len(self.words)

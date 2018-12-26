"""
In dt.py, you will implement a basic decision tree classifier for
binary classification.  Your implementation should be based on the
minimum classification error heuristic (even though this isn't ideal,
it's easier to code than the information-based metrics).
"""

import numpy as np

from binary import *
import util


class DT(BinaryClassifier):
    """
    This class defines the decision tree implementation.  It comes
    with a partial implementation for the tree data structure that
    will enable us to print the tree in a canonical form.
    """

    def __init__(self, opts):
        """
        Initialize our internal state.  The options are:
          opts.maxDepth = maximum number of features to split on
                          (i.e., if maxDepth == 1, then we're a stump)
        """

        self.opts = opts

        # initialize the tree data structure.  all tree nodes have a
        # "isLeaf" field that is true for leaves and false otherwise.
        # leaves have an assigned class (+1 or -1).  internal nodes
        # have a feature to split on, a left child (for when the
        # feature value is < 0.5) and a right child (for when the
        # feature value is >= 0.5)
        
        self.isLeaf = True
        self.label  = 1

    def online(self):
        """
        Our decision trees are batch
        """
        return False

    def __repr__(self):
        """
        Return a string representation of the tree
        """
        return self.displayTree(0)

    def displayTree(self, depth):
        # recursively display a tree
        if self.isLeaf:
            return (" " * (depth*2)) + "Leaf " + repr(self.label) + "\n"
        else:
            if self.opts['criterion'] == 'ig':
                return (" " * (depth*2)) + "Branch " + repr(self.feature) + \
                      " [Gain=" + repr(format(self.gain, '.4f')) + "]\n" + \
                      self.left.displayTree(depth+1) + \
                      self.right.displayTree(depth+1)
            else:
                return (" " * (depth*2)) + "Branch " + repr(self.feature) + \
                      "\n" + self.left.displayTree(depth+1) + \
                      self.right.displayTree(depth+1)

    def predict(self, X):
        """
        Traverse the tree to make predictions for a single sample.  
        You should threshold X at 0.5, so <0.5 means left branch and
        >=0.5 means right branch.
        """

        ### TODO: YOUR CODE HERE
        while(self.isLeaf == False):
            num = self.feature
            if(X[num] < 0.5):
                self = self.left
            else:
                self = self.right
        
        return self.label
        
        


    def trainDT(self, X, Y, maxDepth, criterion, used):
        """
        recursively build the decision tree
        """
   
        # get the size of the data set
        N,D = X.shape

        # check to see if we're either out of depth or no longer
        # have any decisions to make
        if maxDepth <= 0 or np.size(util.uniq(Y)) <= 1:
            # we'd better end at this point.  need to figure
            # out the label to return
            self.isLeaf = True    ### TODO: YOUR CODE HERE
            self.label  = util.mode(Y)  ### TODO: YOUR CODE HERE

        else:
            '''
            - it is very useful to be able to 'split' matrices and vectors:
            if you want the ids for all the Xs for which the 5th feature is
            on, say X(:,5)>=0.5.  If you want the corresponting classes,
            say Y(X(:,5)>=0.5) and if you want the correspnding rows of X,
            say X(X(:,5)>=0.5,:)
            '''
            
            if criterion == 'ig': # information gain
                # compute the entropy at this node
                ### TODO: YOUR CODE HERE                
                posCount = np.sum(Y == 1.0)   
                negCount = np.sum(Y == -1.0)  
                leng = np.size(Y)
                
                posFrac = posCount/leng
                negFrac = negCount/leng
                
                self.entropy = (-(posFrac * math.log((posFrac) ,2))) - (negFrac * math.log((negFrac) ,2))
                
            
            # we need to find a feature to split on
            bestFeature = -1     # which feature has lowest error
            
            # use error stats or gain stats (not both) depending on criterion
            
            # initialize error stats
            bestError  = np.finfo('d').max
            
            # initialize gain stats
            bestGain = np.finfo('d').min         
            
            for d in range(D):
            
                # have we used this feature yet
                if d in used:
                    continue

                # suppose we split on this feature; what labels
                # would go left and right?
                
                on = Y[X[:, d]>=0.5]
                off = Y[X[:, d]<0.5]
                
                leftY  = util.mode(off)     ### TODO: YOUR CODE HERE
                rightY = util.mode(on)    ### TODO: YOUR CODE HERE
                
                # misclassification rate
                if criterion == 'mr':
                    
                    # we'll classify the left points as their most
                    # common class and ditto right points.  our error
                    # is the how many are not their mode.
                    
              
                    leftErr = np.sum(off == (-1 * (leftY)))
                    rightErr = np.sum(on == (-1 * (rightY)))
                    
                   
                    error = leftErr + rightErr    ### TODO: YOUR CODE HERE
                 
                   
                    # update min, max, bestFeature
                    if error <= bestError:
                        bestFeature = d
                        bestError   = error
                    
                    
                # information gain
                elif criterion == 'ig':
             
                 
                    # now use information gain
                    
                    sz = np.size(X[:,d])
     
                    
                    probT = (np.sum(X[:, d]>=0.5))/sz
                    probF = (np.sum(X[:, d]<0.5))/sz  
                    
                    tru = Y[X[:, d]>=0.5]
                    fls = Y[X[:, d]<0.5]
                    
                    tNeg = np.sum(tru == -1.0)
                    tPos = np.sum(tru == 1.0)
                    
                    tSz = np.size(tru)
                    tFrac1 = 0
                    tFrac2 = 0
                    
                    if(tSz != 0):
                        tFrac1 = tNeg/tSz
                        tFrac2 = tPos/tSz
                    
                    tLog1 = 0
                    tLog2 = 0
                    
                    
                    if (tFrac1 > 0):
                        tLog1 = math.log((tFrac1) ,2)
                   
                    if (tFrac2 > 0):
                        tLog2 = math.log((tFrac2) ,2)
                      
                    fNeg = np.sum(fls == -1.0)
                    fPos = np.sum(fls == 1.0)
                    fSz = np.size(fls)
                    fFrac1 = 0
                    fFrac2 = 0
                    
                    if(fSz != 0):
                        fFrac1 = fNeg/fSz
                        fFrac2 = fPos/fSz
                    
                    fLog1 = 0
                    fLog2 = 0
                    
                    if (fFrac1 > 0):
                        fLog1 = math.log((fFrac1) ,2)
                    
                    if (fFrac2 > 0):
                        fLog2 = math.log((fFrac2) ,2)
                 
                    entT = -(tFrac1 * tLog1) - (tFrac2 * tLog2)
                    entF = -(fFrac1 * fLog1) - (fFrac2 * fLog2)
                    
                    gain = self.entropy - ((probT * entT) + (probF * entF))    ### TODO: YOUR CODE HERE
                   
     
  
                   
                    # update min, max, bestFeature
                    if gain >= bestGain:
                        bestFeature = d
                        bestGain = gain
            
            self.gain = bestGain # information gain corresponding to this split
            if bestFeature < 0:
                # this shouldn't happen, but just in case...
                self.isLeaf = True
                self.label  = util.mode(Y)

            else:
                self.isLeaf  = False    ### TODO: YOUR CODE HERE
                self.feature = bestFeature    ### TODO: YOUR CODE HERE
              
                used.append(bestFeature)                              
                                                                                            
                self.left  = DT({'maxDepth': maxDepth-1, 'criterion':criterion})
                self.right = DT({'maxDepth': maxDepth-1, 'criterion':criterion})
              
                # recurse on our children by calling
                #   self.left.trainDT(...) 
                # and
                #   self.right.trainDT(...) 
                # with appropriate arguments
                ### TODO: YOUR CODE HERE
                leftY = Y[X[:,bestFeature]<0.5]
                leftX = X[X[:,bestFeature]<0.5]
                
                rightY = Y[X[:,bestFeature]>=0.5]
                rightX = X[X[:,bestFeature]>=0.5]
                
                self.left.trainDT(leftX, leftY, maxDepth-1, criterion, used)
                self.right.trainDT(rightX, rightY, maxDepth-1, criterion, used)
                                                                                             

    def train(self, X, Y):
        """
        Build a decision tree based on the data from X and Y.  X is a
        matrix (N x D) for N many examples on D features.  Y is an
        N-length vector of +1/-1 entries.

        Some hints/suggestions:
          - make sure you don't build the tree deeper than self.opts['maxDepth']
          
          - make sure you don't try to reuse features (this could lead
            to very deep trees that keep splitting on the same feature
            over and over again)
            
          - it is very useful to be able to 'split' matrices and vectors:
            if you want the ids for all the Xs for which the 5th feature is
            on, say X(:,5)>=0.5.  If you want the corresponting classes,
            say Y(X(:,5)>=0.5) and if you want the correspnding rows of X,
            say X(X(:,5)>=0.5,:)
            
          - i suggest having train() just call a second function that
            takes additional arguments telling us how much more depth we
            have left and what features we've used already

          - take a look at the 'mode' and 'uniq' functions in util.py
        """

        # TODO: implement the function below
        if 'criterion' not in self.opts:
          self.opts['criterion'] = 'mr' # misclassification rate
        self.trainDT(X, Y, self.opts['maxDepth'], self.opts['criterion'], [])


    def getRepresentation(self):
        """
        Return our internal representation: for DTs, this is just our
        tree structure -- i.e., ourselves
        """
        
        return self


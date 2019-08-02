import numpy as np
import functools
import copy
import heapq
import sys
import time

@functools.total_ordering
class puzBoard:
    hFunct = None
    finState = np.zeros((3,3),dtype=np.int8)
    finStateBytes = np.zeros((3, 3), dtype=np.int8).tobytes()
    opList = []
    opListDict = {}
    clListDict = {}
    enc = {'T' + str(i + 1): i + 1 for i in range(8)}
    enc['B'] = 0
    revenc = {i + 1:'T' + str(i + 1)  for i in range(8)}
    revenc[0] = 'B'


    def __init__(self, g, parent, state=np.zeros((3,3),dtype=np.int8)):
        self.state = state
        self.g = g
        self.h = puzBoard.hFunct(self.state)
        self.f = self.g+self.h
        self.bPos = None
        self.parent = parent
        self.getBlankPos()
        self.stateBytes = self.state.tobytes()

    def getBlankPos(self):
        b = np.where(self.state==0)
        c = [b[0][0],b[1][0]]
        self.bPos = c

    def __lt__(self, other):
        if self.f != other.f:
            return self.f < other.f
        else:
            return id(self) < id(other)

    def __eq__(self, other):
        return  id(self) == id(other)

    def getChildren(self):
        # too much repitition.. i know ... just go on with it i guess
        li = []
        if self.bPos[0] != 0: # can go up
            ns = copy.copy(self.state)
            ns[self.bPos[0]][self.bPos[1]] = ns[self.bPos[0]-1][self.bPos[1]]
            ns[self.bPos[0]-1][self.bPos[1]] = 0
            nu = puzBoard(self.g+1,self,ns)
            li.append(nu)
        if self.bPos[0] != 2: # can go down
            ns = copy.copy(self.state)
            ns[self.bPos[0]][self.bPos[1]] = ns[self.bPos[0]+1][self.bPos[1]]
            ns[self.bPos[0]+1][self.bPos[1]] = 0
            nu = puzBoard(self.g+1,self,ns)
            li.append(nu)
        if self.bPos[1] != 2:  # can go right
            ns = copy.copy(self.state)
            ns[self.bPos[0]][self.bPos[1]] = ns[self.bPos[0]][self.bPos[1] +1]
            ns[self.bPos[0]][self.bPos[1]+1] = 0
            nu = puzBoard(self.g + 1, self, ns)
            li.append(nu)
        if self.bPos[1] != 0:  # can go left
            ns = copy.copy(self.state)
            ns[self.bPos[0]][self.bPos[1]] = ns[self.bPos[0]][self.bPos[1]-1]
            ns[self.bPos[0]][self.bPos[1]-1] = 0
            nu = puzBoard(self.g + 1, self, ns)
            li.append(nu)

        return li

    def __str__(self):
        rets = '\n'
        li = []
        for i in self.state:
                li.append('{:^3s} | {:^3s} | {:^3s}'.format(*[puzBoard.revenc[j] for j in i]))

        li.append("G = {} | H = {} | F  = {}".format(self.g,self.h,self.f))
        li.append('='*80)
        li.append('\n')
        return rets.join(li)

    def printPath(self):
        pli = []
        m = self
        while True:
            pli.append(m)
            m = m.parent
            if m is None:
                pli.reverse()
                print('Totally {} States in optimal path'.format(len(pli)))
                for pt in pli:
                    print(pt)
                break



    @staticmethod
    def addNewToList(newState): # add new elements to list
        nsb = newState.stateBytes
        oldState = puzBoard.clListDict.get(nsb,None)
        if oldState is None: # not present in closedList
            oldState = puzBoard.opListDict.get(nsb,None)
            if oldState is None: # not present in openlist
                heapq.heappush(puzBoard.opList,newState)
                puzBoard.opListDict[nsb] = newState
                # print(nsb,newState.stateBytes, nsb == newState.stateBytes)
            else: # present in openlist
                if oldState.f > newState.f:
                    puzBoard.opList.remove(oldState)
                    puzBoard.opList.append(newState)
                    heapq.heapify(puzBoard.opList)
                    del puzBoard.opListDict[nsb]
                    puzBoard.opListDict[nsb] = newState
                    # print(nsb, newState.stateBytes, nsb == newState.stateBytes)
        # else: # present in closedList
        #     if oldState > newState:
        #         del puzBoard.clListDict[nsb]
        #         heapq.heappush(puzBoard.opList,newState)
        #         # heapq.heapify(puzBoard.opList)
        #         puzBoard.opListDict[nsb] = newState

    @staticmethod
    def addToClList(newState):
        nsb = newState.stateBytes
        # del puzBoard.opListDict[nsb]
        puzBoard.clListDict[nsb] = newState

    @classmethod
    def h1(cls,st):
        return 0

    @classmethod
    def h2(cls,st):
        d = (np.equal(st,cls.finState)*1).sum()
        return 9-d

    @classmethod
    def h3(cls,st):
        ans = 0
        for i in range(3):
            for j in range(3):
                c = np.where(st == cls.finState[i][j])
                ans += abs(c[0] - i)
                ans += abs(c[1] - j)
        return int(ans)

    @classmethod
    def h4(cls,st):
        return 362880 # 9! factorial(9)


def AStar(strt, heur):
    heurOptions = {0:(puzBoard.h1,'All Zeros'),
                   1:(puzBoard.h2,' # Displaced'),
                   2:(puzBoard.h3,'Manhattan Distance'),
                   3:(puzBoard.h4, '9 Factorial')}
                

    puzBoard.finState = strt[3:]
    puzBoard.finStateBytes = puzBoard.finState.tobytes()
    puzBoard.hFunct = heurOptions[heur][0]

    s = puzBoard(0,None,strt[:3])
    puzBoard.addNewToList(s)
    print('Using heurestic:{}\nStart State: '.format(heurOptions[heur][1]))
    print(s)

    print('Goal State: ')
    print(puzBoard(0,None,strt[3:]))
    start_time = time.time()
    end_time = None
    while True:
        cn = heapq.heappop(puzBoard.opList)
        del puzBoard.opListDict[cn.stateBytes]
        puzBoard.addToClList(cn)

        if cn.stateBytes == puzBoard.finStateBytes:
            end_time = time.time()
            print('Success.  Optimal Path of {} length found after exploring {} number of Elements'.format(cn.g,
                                                                                                           len(puzBoard.clListDict)) )
            # print(cn)
            cn.printPath()
            break
        cnChildren = cn.getChildren()
        for ns in cnChildren:
            puzBoard.addNewToList(ns)

        # del puzBoard.opListDict[cn.state.tobytes()]

        if len(puzBoard.opList) == 0:
            end_time = time.time()
            print('Failure :(  Explored {} States'.format(len(puzBoard.clListDict)))
            break

    print('Total Execution Time: {} sec'.format(end_time-start_time))


if __name__ == '__main__':
    fname = sys.argv[1] # commandline arguement : path to input file

    with open(fname,'r') as f:
        inp = f.readlines()

    strt = np.zeros((6, 3), dtype=np.int8)
    # fin = np.zeros((3, 3), dtype=np.int8)
    p = 0
    for lin in inp:
        a = lin.strip().split(' ')
        if len(a) == 0:
            continue
        a = np.asarray([puzBoard.enc[m] for m in a], dtype=np.int8)
        a = a[:3]
        strt[p] += a
        p += 1

    AStar(strt,0)

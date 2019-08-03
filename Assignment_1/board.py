import numpy
import functools
import copy
import heapq
import sys
import time
import matplotlib.pyplot as plt


total_expanded_states = ['Total # States Expanded']
optimal_path_length = ['# States in Optimal Path']
optimal_path = ['Optimal Path Actions']
optimal_cost = ['Optimal Cost']
execution_time = ['Execution Time']


@functools.total_ordering
class PuzzleBoard:

    heuristic_function = None
    final_state = numpy.zeros((3, 3), dtype = numpy.int8)
    final_state_bytes = numpy.zeros((3, 3), dtype = numpy.int8).tobytes()
    open_list = []
    open_list_dict = {}
    closed_list_dict = {}
    encoding = {'T' + str(i + 1) : i + 1 for i in range(8)}
    encoding['B'] = 0
    decoding = {i + 1 : 'T' + str(i + 1)  for i in range(8)}
    decoding[0] = 'B'

    def __init__(self, g, parent, state = numpy.zeros((3, 3), dtype = numpy.int8)):
        self.state = state
        self.g = g
        self.h = PuzzleBoard.heuristic_function(self.state)
        self.f = self.g + self.h
        self.blank_position = None
        self.parent = parent
        self.get_blank_position()
        self.state_bytes = self.state.tobytes()

    def get_blank_position(self):
        b = numpy.where(self.state == 0)
        c = [b[0][0], b[1][0]]
        self.blank_position = c

    def __lt__(self, other):
        if self.f != other.f:
            return self.f < other.f
        else:
            return id(self) < id(other)

    def __eq__(self, other):
        return  id(self) == id(other)

    def get_children(self):
        li = []

        if self.blank_position[0] != 0: # can go up
            new_state = copy.copy(self.state)
            new_state[self.blank_position[0]][self.blank_position[1]] = new_state[self.blank_position[0] - 1][self.blank_position[1]]
            new_state[self.blank_position[0] - 1][self.blank_position[1]] = 0
            nu = PuzzleBoard(self.g + 1, self, new_state)
            li.append(nu)

        if self.blank_position[0] != 2: # can go down
            new_state = copy.copy(self.state)
            new_state[self.blank_position[0]][self.blank_position[1]] = new_state[self.blank_position[0] + 1][self.blank_position[1]]
            new_state[self.blank_position[0] + 1][self.blank_position[1]] = 0
            nu = PuzzleBoard(self.g + 1, self, new_state)
            li.append(nu)

        if self.blank_position[1] != 2:  # can go right
            new_state = copy.copy(self.state)
            new_state[self.blank_position[0]][self.blank_position[1]] = new_state[self.blank_position[0]][self.blank_position[1] + 1]
            new_state[self.blank_position[0]][self.blank_position[1] + 1] = 0
            nu = PuzzleBoard(self.g + 1, self, new_state)
            li.append(nu)

        if self.blank_position[1] != 0:  # can go left
            new_state = copy.copy(self.state)
            new_state[self.blank_position[0]][self.blank_position[1]] = new_state[self.blank_position[0]][self.blank_position[1] - 1]
            new_state[self.blank_position[0]][self.blank_position[1] - 1] = 0
            nu = PuzzleBoard(self.g + 1, self, new_state)
            li.append(nu)

        return li

    def __str__(self):
        rets = '\n'
        li = []

        for i in self.state:
            li.append('{:^3s} | {:^3s} | {:^3s}'.format(*[PuzzleBoard.decoding[j] for j in i]))

        li.append("G = {} | H = {} | F  = {}".format(self.g, self.h, self.f))
        li.append('=' * 80)
        li.append('\n')

        return rets.join(li)

    def print_path(self):
        pli = []
        m = self

        while True:
            pli.append(m)
            m = m.parent

            if m is None:
                pli.reverse()
                print('Totally {} States In Optimal Path.'.format(len(pli)))

                for pt in pli:
                    print(pt)

                break

        actList = []

        for i in range(len(pli) - 1):
            f1 = numpy.where(pli[i].state == 0)
            f2 = numpy.where(pli[i + 1].state == 0)

            if f1[0][0] == f2[0][0]:
                if f1[1][0] > f2[1][0]: #right
                    actList.append('R')
                else:
                    actList.append('L')
            else:
                if f1[0][0] > f2[0][0]: # Down
                    actList.append('D')
                else:
                    actList.append('U')

        retf = ''.join(actList)
        print('Actions Taken: ', retf)
        return retf

    @staticmethod
    def add_new_to_list(new_state): # add new elements to list
        new_state_bytes = new_state.state_bytes
        oldState = PuzzleBoard.closed_list_dict.get(new_state_bytes, None)

        if oldState is None: # not present in closedList
            oldState = PuzzleBoard.open_list_dict.get(new_state_bytes, None)

            if oldState is None: # not present in openlist
                heapq.heappush(PuzzleBoard.open_list, new_state)
                PuzzleBoard.open_list_dict[new_state_bytes] = new_state
            else: # present in openlist
                if oldState.f > new_state.f:
                    PuzzleBoard.open_list.remove(oldState)
                    PuzzleBoard.open_list.append(new_state)
                    heapq.heapify(PuzzleBoard.open_list)
                    del PuzzleBoard.open_list_dict[new_state_bytes]
                    PuzzleBoard.open_list_dict[new_state_bytes] = new_state

    @staticmethod
    def add_to_closed_list(new_state):
        new_state_bytes = new_state.state_bytes
        PuzzleBoard.closed_list_dict[new_state_bytes] = new_state

    @classmethod
    def h1(cls, state):
        return 0

    @classmethod
    def h2(cls, state):
        d = (numpy.equal(state, cls.final_state) * 1).sum()
        return 9 - d

    @classmethod
    def h3(cls, state):
        ans = 0

        for i in range(3):
            for j in range(3):
                c = numpy.where(state == cls.final_state[i][j])
                ans += abs(c[0] - i)
                ans += abs(c[1] - j)

        return int(ans)

    @classmethod
    def h4(cls, state):
        # 9! = factorial(9)
        return 362880

    @classmethod
    def reset_lists(cls):
        cls.heuristic_function = None
        cls.final_state = numpy.zeros((3, 3), dtype = numpy.int8)
        cls.final_state_bytes = numpy.zeros((3, 3), dtype = numpy.int8).tobytes()
        cls.open_list = []
        cls.open_list_dict = {}
        cls.closed_list_dict = {}


def AStar(start, heuristic):
    heuristic_options = {0 : (PuzzleBoard.h1, 'All Zeroes'),
                         1 : (PuzzleBoard.h2, '# Tiles Displaced'),
                         2 : (PuzzleBoard.h3, 'Manhattan Distance'),
                         3 : (PuzzleBoard.h4, '9 Factorial')}
    PuzzleBoard.final_state = start[3 :]
    PuzzleBoard.final_state_bytes = PuzzleBoard.final_state.tobytes()
    PuzzleBoard.heuristic_function = heuristic_options[heuristic][0]

    s = PuzzleBoard(0, None, start[: 3])
    PuzzleBoard.add_new_to_list(s)
    print('Using Heuristic: {}\nStart State: '.format(heuristic_options[heuristic][1]))
    print(s)

    print('Goal State: ')
    print(PuzzleBoard(0, None, start[3 :]))
    start_time = time.time()
    end_time = None

    while True:
        current = heapq.heappop(PuzzleBoard.open_list)
        del PuzzleBoard.open_list_dict[current.state_bytes]
        PuzzleBoard.add_to_closed_list(current)

        if current.state_bytes == PuzzleBoard.final_state_bytes:
            end_time = time.time()
            print('Success. Optimal Path of {} Length Found After Exploring {} Elements'.format(current.g, len(PuzzleBoard.closed_list_dict)))
            total_expanded_states.append(len(PuzzleBoard.closed_list_dict))
            optimal_path_length.append(current.g + 1)
            optimal_path.append(current.print_path())
            optimal_cost.append(current.g)
            break

        current_children = current.get_children()

        for new_state in current_children:
            PuzzleBoard.add_new_to_list(new_state)

        if len(PuzzleBoard.open_list) == 0:
            end_time = time.time()
            print('Failure :(  Explored {} States.'.format(len(PuzzleBoard.closed_list_dict)))
            total_expanded_states.append(len(PuzzleBoard.closed_list_dict))
            optimal_path_length.append(float('nan'))
            optimal_path.append(float('nan'))
            optimal_cost.append(float('nan'))
            break

    print('Total Execution Time: {} seconds'.format(end_time - start_time))
    execution_time.append(float(end_time - start_time))


def my_plot(vaList, filename, yLabel):
    plt.xticks([1, 2, 3, 4], ['h1: All Zeroes', 'h2: Tiles Displaced',
                              'h3: Manhattan Distance', 'h4: 9 Factorial'])
    plt.plot([1, 2, 3, 4], vaList[1 :])
    plt.xlabel('Heuristics')
    plt.ylabel(yLabel)
    plt.savefig(filename, bbox_inches = 'tight')
    plt.clf()


if __name__ == '__main__':
    if len(sys.argv) == 0:
        print('Please provide input file as command-line argument.')
        exit()

    # Command-line argument: Path to input file.
    fname = sys.argv[1]

    with open(fname,'r') as f:
        inp = f.readlines()

    start = numpy.zeros((6, 3), dtype = numpy.int8)
    p = 0

    for lin in inp:
        a = lin.strip().split(' ')

        if len(a) == 0:
            continue

        a = numpy.asarray([PuzzleBoard.encoding[m] for m in a], dtype = numpy.int8)[: 3]
        start[p] += a
        p += 1

    for i in range(4):
        PuzzleBoard.reset_lists()
        AStar(start, i)

    with open('Table.csv', 'a') as f:
        f.write('Heuristic:, All Zeroes, # Tiles Displaced, Manhattan Distance, 9 Factorial\n')
        f.write('%s, %0.4f, %0.4f, %0.4f, %0.4f\n' % (*total_expanded_states, ))
        f.write('%s, %0.4f, %0.4f, %0.4f, %0.4f\n' % (*optimal_path_length, ))
        f.write('%s, %s, %s, %s, %s\n' % (*optimal_path, ))
        f.write('%s, %0.4f, %0.4f, %0.4f, %0.4f\n' % (*optimal_cost, ))
        f.write('%s, %0.4f, %0.4f, %0.4f, %0.4f\n' % (*execution_time, ))

    my_plot(total_expanded_states, 'ExploredStates.png', 'Number of Expanded States')
    my_plot(execution_time, 'ExecutionTime.png', 'Time (seconds)')

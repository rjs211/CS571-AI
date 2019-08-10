import numpy
import functools
import copy
import heapq
import sys
import time
import matplotlib.pyplot as plt


# Initial lists.
total_expanded_states = ['Total # States Expanded']
optimal_path_length = ['# States in Optimal Path']
optimal_path = ['Optimal Path Actions']
optimal_cost = ['Optimal Cost']
execution_time = ['Execution Time']
all_expanded_states = ['All Expanded States']
is_monotone = ['Monotonicity']


@functools.total_ordering
class PuzzleBoard:
    """
    Class for emulating the search space for the A* algorithm.
    """
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
        """
        Function for initialising the PuzzleBoard class.
        """
        self.state = state
        self.g = g
        self.h = PuzzleBoard.heuristic_function(self.state)
        self.f = self.g + self.h
        self.blank_position = None
        self.parent = parent
        self.get_blank_position()
        self.state_bytes = self.state.tobytes()

    def get_blank_position(self):
        """
        Function for obtaining the blank position from the state.
        """
        b = numpy.where(self.state == 0)
        c = [b[0][0], b[1][0]]
        self.blank_position = c

    def __lt__(self, other):
        """
        Function for overloading the less than operator.
        """
        if self.f != other.f:
            return self.f < other.f
        else:
            return id(self) < id(other)

    def __eq__(self, other):
        """
        Function for overloading the equality operator.
        """
        return  id(self) == id(other)

    def get_children(self):
        """
        Function for getting the child states from the current state.
        """
        current_list = []
        current_monotone = True

        # Check for upward movement.
        if self.blank_position[0] != 0:
            new_state = copy.copy(self.state)
            new_state[self.blank_position[0]][self.blank_position[1]] = new_state[self.blank_position[0] - 1][self.blank_position[1]]
            new_state[self.blank_position[0] - 1][self.blank_position[1]] = 0
            nu = PuzzleBoard(self.g + 1, self, new_state)
            current_list.append(nu)
            if self.h - nu.h > 1:
                current_monotone = False

        # Check for downward movement.
        if self.blank_position[0] != 2:
            new_state = copy.copy(self.state)
            new_state[self.blank_position[0]][self.blank_position[1]] = new_state[self.blank_position[0] + 1][self.blank_position[1]]
            new_state[self.blank_position[0] + 1][self.blank_position[1]] = 0
            nu = PuzzleBoard(self.g + 1, self, new_state)
            current_list.append(nu)
            if self.h - nu.h > 1:
                current_monotone = False

        # Check for right movement.
        if self.blank_position[1] != 2:
            new_state = copy.copy(self.state)
            new_state[self.blank_position[0]][self.blank_position[1]] = new_state[self.blank_position[0]][self.blank_position[1] + 1]
            new_state[self.blank_position[0]][self.blank_position[1] + 1] = 0
            nu = PuzzleBoard(self.g + 1, self, new_state)
            current_list.append(nu)
            if self.h - nu.h > 1:
                current_monotone = False

        # Check for left movement.
        if self.blank_position[1] != 0:
            new_state = copy.copy(self.state)
            new_state[self.blank_position[0]][self.blank_position[1]] = new_state[self.blank_position[0]][self.blank_position[1] - 1]
            new_state[self.blank_position[0]][self.blank_position[1] - 1] = 0
            nu = PuzzleBoard(self.g + 1, self, new_state)
            current_list.append(nu)
            if self.h - nu.h > 1:
                current_monotone = False

        return current_list, current_monotone

    def __str__(self):
        """
        Function for converting the current state to string format.
        """
        return_string = '\n'
        current_list = []

        for i in self.state:
            current_list.append('{:^3s} | {:^3s} | {:^3s}'.format(*[PuzzleBoard.decoding[j] for j in i]))

        current_list.append("G = {} | H = {} | F  = {}".format(self.g, self.h, self.f))
        current_list.append('=' * 80)
        current_list.append('\n')

        return return_string.join(current_list)

    def print_path(self):
        """
        Function for printing the optimal path explored.
        """
        path_list = []
        m = self

        while True:
            path_list.append(m)
            m = m.parent

            if m is None:
                path_list.reverse()
                print('Totally {} States In Optimal Path.'.format(len(path_list)))

                for path in path_list:
                    print(path)

                break

        action_list = []

        for i in range(len(path_list) - 1):
            f1 = numpy.where(path_list[i].state == 0)
            f2 = numpy.where(path_list[i + 1].state == 0)

            if f1[0][0] == f2[0][0]:
                if f1[1][0] > f2[1][0]: #right
                    action_list.append('R')
                else:
                    action_list.append('L')
            else:
                if f1[0][0] > f2[0][0]: # Down
                    action_list.append('D')
                else:
                    action_list.append('U')

        retf = ''.join(action_list)
        print('Actions Taken: ', retf)
        return retf

    @staticmethod
    def add_to_open_list(new_state):
        """
        Function for adding new states to the open list.
        """
        new_state_bytes = new_state.state_bytes
        old_state = PuzzleBoard.closed_list_dict.get(new_state_bytes, None)

        if old_state is None: # not present in closedList
            old_state = PuzzleBoard.open_list_dict.get(new_state_bytes, None)

            if old_state is None: # not present in openlist
                heapq.heappush(PuzzleBoard.open_list, new_state)
                PuzzleBoard.open_list_dict[new_state_bytes] = new_state
            else: # present in openlist
                if old_state.f > new_state.f:
                    PuzzleBoard.open_list.remove(old_state)
                    PuzzleBoard.open_list.append(new_state)
                    heapq.heapify(PuzzleBoard.open_list)
                    del PuzzleBoard.open_list_dict[new_state_bytes]
                    PuzzleBoard.open_list_dict[new_state_bytes] = new_state

    @staticmethod
    def add_to_closed_list(new_state):
        """
        Function for adding new states to the closed list.
        """
        new_state_bytes = new_state.state_bytes
        PuzzleBoard.closed_list_dict[new_state_bytes] = new_state

    @classmethod
    def h1(cls, state):
        """
        Function for all zero heuristic.
        """
        return 0

    @classmethod
    def h2(cls, state):
        """
        Function for displaced tiles heuristic.
        """
        d = (numpy.equal(state, cls.final_state) * 1).sum()
        if not numpy.all(numpy.where(state.reshape(-1) == 0)[0] == numpy.where(cls.final_state.reshape(-1) == 0)[0]):
            pass
        return 9 - d

    @classmethod
    def h2_no_blank(cls, state):
        """
        Function for displaced tiles heuristic, without counting the blank state
        """
        d = (numpy.equal(state, cls.final_state) * 1).sum()
        if not numpy.all(numpy.where(state.reshape(-1) == 0)[0] == numpy.where(cls.final_state.reshape(-1) == 0)[0]):
            d += 1
        return 9 - d

    @classmethod
    def h3(cls, state):
        """
        Function for Manhattan Distance heuristic.
        """
        ans = 0

        for i in range(3):
            for j in range(3):
                if cls.final_state[i][j] == 0:
                    pass
                c = numpy.where(state == cls.final_state[i][j])
                ans += abs(c[0] - i)
                ans += abs(c[1] - j)

        return int(ans)

    @classmethod
    def h3_no_blank(cls, state):
        """
        Function for Manhattan Distance heuristic, without counting the blank state
        """
        ans = 0

        for i in range(3):
            for j in range(3):
                if cls.final_state[i][j] == 0:
                    continue
                c = numpy.where(state == cls.final_state[i][j])
                ans += abs(c[0] - i)
                ans += abs(c[1] - j)

        return int(ans)
    
    @classmethod
    def h4(cls, state):
        """
        Function for factorial of 9 heuristic.
        """
        return 362880

    @classmethod
    def reset_lists(cls):
        """
        Function for resetting all the lists.
        """
        cls.heuristic_function = None
        cls.final_state = numpy.zeros((3, 3), dtype = numpy.int8)
        cls.final_state_bytes = numpy.zeros((3, 3), dtype = numpy.int8).tobytes()
        cls.open_list = []
        cls.open_list_dict = {}
        cls.closed_list_dict = {}


def AStar(start, heuristic):
    """
    Function for running the A* algorithm given the heuristics.
    """
    heuristic_options = {0 : (PuzzleBoard.h1, 'All Zeroes'),
                         1 : (PuzzleBoard.h2, '# Tiles Displaced - Counting Blank'),
                         2 : (PuzzleBoard.h3, 'Manhattan Distance - Counting Blank'),
                         3 : (PuzzleBoard.h4, '9 Factorial'),
                         4 : (PuzzleBoard.h2_no_blank, '# Tiles Displaced - No Blank'),
                         5 : (PuzzleBoard.h3_no_blank, 'Manhattan Distance - No Blank'),}
    PuzzleBoard.final_state = start[3 :]
    PuzzleBoard.final_state_bytes = PuzzleBoard.final_state.tobytes()
    PuzzleBoard.heuristic_function = heuristic_options[heuristic][0]

    s = PuzzleBoard(0, None, start[: 3])
    PuzzleBoard.add_to_open_list(s)
    print('Using Heuristic: {}\nStart State: '.format(heuristic_options[heuristic][1]))
    print(s)

    print('Goal State: ')
    print(PuzzleBoard(0, None, start[3 :]))
    start_time = time.time()
    end_time = None
    heuristic_monotone = True

    while True:
        current = heapq.heappop(PuzzleBoard.open_list)
        del PuzzleBoard.open_list_dict[current.state_bytes]
        PuzzleBoard.add_to_closed_list(current)

        if current.state_bytes == PuzzleBoard.final_state_bytes:
            end_time = time.time()
            print('Success. Optimal Path of {} Length Found After Exploring {} Elements'.format(current.g, len(PuzzleBoard.closed_list_dict)))
            total_expanded_states.append(len(PuzzleBoard.closed_list_dict))
            all_expanded_states.append(PuzzleBoard.closed_list_dict.keys())
            optimal_path_length.append(current.g + 1)
            optimal_path.append(current.print_path())
            optimal_cost.append(current.g)
            break

        current_children, current_monotone = current.get_children()
        heuristic_monotone = heuristic_monotone and current_monotone

        for new_state in current_children:
            PuzzleBoard.add_to_open_list(new_state)

        if len(PuzzleBoard.open_list) == 0:
            end_time = time.time()
            print('Failure :(  Explored {} States.'.format(len(PuzzleBoard.closed_list_dict)))
            total_expanded_states.append(len(PuzzleBoard.closed_list_dict))
            all_expanded_states.append(PuzzleBoard.closed_list_dict.keys())
            optimal_path_length.append(float('nan'))
            optimal_path.append(float('nan'))
            optimal_cost.append(float('nan'))
            break

    print('Total Execution Time: {} seconds'.format(end_time - start_time))
    execution_time.append(float(end_time - start_time))
    is_monotone.append(heuristic_monotone)


def my_plot(values, filename, y_label):
    """
    Function for plotting the number of explored states and time graphs.
    """
    plt.xticks([1, 2, 3, 4, 5, 6],
               ['h1: All Zeroes', 'h2: Tiles Displaced - Counting Blank',
                'h3: Manhattan Distance - Counting Blank', 'h4: 9 Factorial',
                'h5: Tiles Displaced - No Blank',
                'h6: Manhattan Distance - No Blank'], rotation = 90)
    plt.plot([1, 2, 3, 4, 5, 6], values[1 :])
    plt.xlabel('Heuristics')
    plt.ylabel(y_label)
    plt.savefig(filename, bbox_inches = 'tight')
    plt.clf()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        fname = '../Assignment_2/inp.txt'
    else:
        # Command-line argument: Path to input file.
        fname = sys.argv[1]

    with open(fname, 'r') as f:
        input_lines = f.readlines()

    start = numpy.zeros((6, 3), dtype = numpy.int8)
    p = 0

    for line in input_lines:
        a = line.strip().split(' ')

        if len(a) == 0:
            continue

        a = numpy.asarray([PuzzleBoard.encoding[m] for m in a], dtype = numpy.int8)[: 3]
        start[p] += a
        p += 1

    for i in range(6):
        PuzzleBoard.reset_lists()
        AStar(start, i)

    with open('Table.csv', 'a') as f:
        f.write('Heuristic:, All Zeroes, # Tiles Displaced - Counting Blank, Manhattan Distance - Counting Blank, 9 Factorial, # Tiles Displaced - No Blank, Manhattan Distance - No Blank\n')
        f.write('%s, %0.4f, %0.4f, %0.4f, %0.4f, %0.4f, %0.4f\n' % (*total_expanded_states, ))
        f.write('%s, %0.4f, %0.4f, %0.4f, %0.4f, %0.4f, %0.4f\n' % (*optimal_path_length, ))
        f.write('%s, %s, %s, %s, %s, %s, %s\n' % (*optimal_path, ))
        f.write('%s, %0.4f, %0.4f, %0.4f, %0.4f, %0.4f, %0.4f\n' % (*optimal_cost, ))
        f.write('%s, %0.4f, %0.4f, %0.4f, %0.4f, %0.4f, %0.4f\n' % (*execution_time, ))
        f.write('%s, %s, %s, %s, %s, %s, %s\n' % (*is_monotone, ))

    my_plot(total_expanded_states, 'ExploredStates.png', 'Number of Expanded States')
    my_plot(execution_time, 'ExecutionTime.png', 'Time (seconds)')

    # Comparing states explored by better heuristics.
    to_compare = [(1, 5), (1, 6), (5, 6)]
    heuristic_names = 'Heuristic:, All Zeroes, # Tiles Displaced - Counting Blank, Manhattan Distance - Counting Blank, 9 Factorial, # Tiles Displaced - No Blank, Manhattan Distance - No Blank'.split(', ')
    for worse_h, better_h in to_compare:
        if not (set(all_expanded_states[better_h]) - set(all_expanded_states[worse_h])):
            print("'{}' expanded all the states expanded by '{}'.".format(heuristic_names[worse_h], heuristic_names[better_h]))
        else:
            print("'{}' did not expand all the states expanded by '{}'.".format(heuristic_names[worse_h], heuristic_names[better_h]))

import copy


class SudokuPuzzleSolver:
    """
    A class to solve a Sudoku puzzle.

    Attributes:
    -----------
    puzzle: List[List[int]]
        The original Sudoku puzzle provided for solving.
    solution: List[List[int]]
        The solved version of the Sudoku puzzle.

    Methods:
    --------
    solve_puzzle(puzzle): Solves the given Sudoku puzzle.
    stringify(puzzle): Converts the puzzle to a formatted string.
    """

    GRID_SIZE = 9
    SUBGRID_SIZE = 3

    def __init__(self, sudoku_puzzle=None):
        """ Initialize the solver with the given puzzle. """
        self.puzzle = sudoku_puzzle
        self.solution = None
        self._possible_values = None
        self._possible_value_changes = None
        if sudoku_puzzle:
            self.solve_puzzle(sudoku_puzzle)

    def solve_puzzle(self, sudoku_puzzle):
        """ Solves the given Sudoku puzzle. """
        self.puzzle = sudoku_puzzle
        self.solution = copy.deepcopy(sudoku_puzzle)
        self._possible_values = self._get_initial_possible_values()
        self._possible_value_changes = []
        self._solve_puzzle()

    @staticmethod
    def stringify(sudoku_puzzle):
        """ Convert the given Sudoku puzzle into a formatted string representation and return it. """
        lines = []
        for i, row in enumerate(sudoku_puzzle):
            if i % SudokuPuzzleSolver.SUBGRID_SIZE == 0 and i != 0:
                lines.append('-' * 21)
            line = []
            for j, num in enumerate(row):
                if j % SudokuPuzzleSolver.SUBGRID_SIZE == 0 and j != 0:
                    line.append('|')
                line.append(str(num) if num != 0 else ' ')
            lines.append(' '.join(line))
        return '\n'.join(lines)

    def _solve_puzzle(self):
        """ Recursive backtracking method to solve the puzzle. Guided by heuristics. """
        cell = self._select_variable()
        if cell is None:  # if there are no empty cells, the puzzle is solved
            return True

        row, col = cell
        sorted_possible_values = sorted(self._possible_values[row][col],
                                        key=lambda x: self._value_constraint_count(x, row, col),
                                        reverse=True)

        for val in sorted_possible_values:
            if self._forward_check(row, col, val):
                self._push_changes_onto_stack(row, col, val)
                self.solution[row][col] = val
                self._update_possible_values(row, col, val)

                if self._solve_puzzle():
                    return True

                self.solution[row][col] = 0
                self._pop_changes_from_stack()

        return False

    @staticmethod
    def _get_block_start(row, col):
        """ Calculates the starting row and column of the block containing the cell. """
        return row // SudokuPuzzleSolver.SUBGRID_SIZE * SudokuPuzzleSolver.SUBGRID_SIZE, col // SudokuPuzzleSolver.SUBGRID_SIZE * SudokuPuzzleSolver.SUBGRID_SIZE

    def _get_initial_possible_values(self):
        """ Initializes possible values based on the given puzzle. """
        self._possible_values = [[set(range(1, 10)) for _ in range(SudokuPuzzleSolver.GRID_SIZE)] for _ in range(SudokuPuzzleSolver.GRID_SIZE)]
        for row in range(SudokuPuzzleSolver.GRID_SIZE):
            for col in range(SudokuPuzzleSolver.GRID_SIZE):
                if self.puzzle[row][col]:
                    self._update_possible_values(row, col, self.puzzle[row][col])
        return self._possible_values

    def _update_possible_values(self, row, col, val):
        """ Update the possible values after assigning a value to a cell. """
        self._possible_values[row][col] = {val}

        for i in range(SudokuPuzzleSolver.GRID_SIZE):
            if self.solution[i][col] == 0:
                self._possible_values[i][col].discard(val)
            if self.solution[row][i] == 0:
                self._possible_values[row][i].discard(val)

        block_row_start, block_col_start = self._get_block_start(row, col)
        for i in range(SudokuPuzzleSolver.SUBGRID_SIZE):
            for j in range(SudokuPuzzleSolver.SUBGRID_SIZE):
                if self.solution[block_row_start + i][block_col_start + j] == 0:
                    self._possible_values[block_row_start + i][block_col_start + j].discard(val)

    def _push_changes_onto_stack(self, row, col, val):
        """ Push the current state changes onto the stack. """
        changes = {'cell': (row, col), 'value': val, 'affected': []}

        for i in range(SudokuPuzzleSolver.GRID_SIZE):
            if val in self._possible_values[row][i]:
                changes['affected'].append((row, i, copy.copy(self._possible_values[row][i])))
            if val in self._possible_values[i][col]:
                changes['affected'].append((i, col, copy.copy(self._possible_values[i][col])))

        start_row, start_col = self._get_block_start(row, col)
        for i in range(SudokuPuzzleSolver.SUBGRID_SIZE):
            for j in range(SudokuPuzzleSolver.SUBGRID_SIZE):
                r, c = start_row + i, start_col + j
                if val in self._possible_values[r][c]:
                    changes['affected'].append((r, c, copy.copy(self._possible_values[r][c])))

        self._possible_value_changes.append(changes)

    def _pop_changes_from_stack(self):
        """ Revert the changes by popping them from the stack. """
        changes = self._possible_value_changes.pop()
        row, col = changes['cell']
        val = changes['value']

        self._possible_values[row][col].add(val)

        for r, c, possible_values in changes['affected']:
            self._possible_values[r][c] = possible_values

    def _forward_check(self, row, col, val):
        """ Check whether a variable may be assigned to a cell without eliminating all possible assignments for another cell. """
        for i in range(SudokuPuzzleSolver.GRID_SIZE):
            if i != col and len(self._possible_values[row][i]) == 1 and val in self._possible_values[row][i]:
                return False
            if i != row and len(self._possible_values[i][col]) == 1 and val in self._possible_values[i][col]:
                return False

        block_row_start, block_col_start = self._get_block_start(row, col)
        for i in range(SudokuPuzzleSolver.SUBGRID_SIZE):
            for j in range(SudokuPuzzleSolver.SUBGRID_SIZE):
                r, c = block_row_start + i, block_col_start + j
                if len(self._possible_values[r][c]) == 1 and val in self._possible_values[r][c] and (r, c) != (row, col):
                    return False

        return True

    def _select_variable(self):
        """ Select a cell using the most constrained and most constraining variable heuristics. """
        most_constrained = None
        min_constraint = float('inf')
        max_constraining = -1

        for row in range(SudokuPuzzleSolver.GRID_SIZE):
            for col in range(SudokuPuzzleSolver.GRID_SIZE):
                if self.solution[row][col] == 0:
                    constraint = self._variable_constraint_count((row, col))
                    constraining = self._variable_constraining_count((row, col))

                    if constraint < min_constraint or (
                            constraint == min_constraint and constraining > max_constraining):
                        most_constrained = (row, col)
                        min_constraint = constraint
                        max_constraining = constraining

        return most_constrained

    def _variable_constraint_count(self, cell):
        """ Calculate a measure of the variables constraints,
            to be used to determine the most constrained variable
            (variable with the fewest possible values).
        """
        row, col = cell
        return len(self._possible_values[row][col])

    def _variable_constraining_count(self, cell):
        """ Calculate a measure of the variables constraining power,
            to be used to determine the most constraining variable
            (variable that constrains the greatest number of other variables).
        """
        row, col = cell
        block_row_start, block_col_start = self._get_block_start(row, col)

        empty_cells_in_row = sum(1 for i in range(SudokuPuzzleSolver.GRID_SIZE) if self.solution[row][i] == 0)
        empty_cells_in_col = sum(1 for i in range(SudokuPuzzleSolver.GRID_SIZE) if self.solution[i][col] == 0)
        empty_cells_in_box = sum(1 for i in range(SudokuPuzzleSolver.SUBGRID_SIZE) for j in range(SudokuPuzzleSolver.SUBGRID_SIZE) if self.solution[block_row_start + i][block_col_start + j] == 0)

        return empty_cells_in_row + empty_cells_in_col + empty_cells_in_box

    def _value_constraint_count(self, val, row, col):
        """ Calculate the number of remaining choices in the affected cells if the value is placed in the given cell,
            to be used to determine the least constraining value
            (value that leaves the most choices for remaining variables).
        """
        count = 0
        for i in range(SudokuPuzzleSolver.GRID_SIZE):
            for j in range(SudokuPuzzleSolver.GRID_SIZE):
                in_same_box = i // SudokuPuzzleSolver.SUBGRID_SIZE == row // SudokuPuzzleSolver.SUBGRID_SIZE and j // SudokuPuzzleSolver.SUBGRID_SIZE == col // SudokuPuzzleSolver.SUBGRID_SIZE
                if self.solution[i][j] == 0 and (i == row or j == col or in_same_box):
                    temp_possible_values = self._possible_values[i][j].copy()
                    temp_possible_values.discard(val)
                    count += len(temp_possible_values)
        return count


if __name__ == "__main__":
    # Example usage
    puzzle = [[5, 3, 0, 0, 7, 0, 0, 0, 0],
              [6, 0, 0, 1, 9, 5, 0, 0, 0],
              [0, 9, 8, 0, 0, 0, 0, 6, 0],
              [8, 0, 0, 0, 6, 0, 0, 0, 3],
              [4, 0, 0, 8, 0, 3, 0, 0, 1],
              [7, 0, 0, 0, 2, 0, 0, 0, 6],
              [0, 6, 0, 0, 0, 0, 2, 8, 0],
              [0, 0, 0, 4, 1, 9, 0, 0, 5],
              [0, 0, 0, 0, 8, 0, 0, 7, 9]]

    solver = SudokuPuzzleSolver(puzzle)
    print("Solved Sudoku:")
    print(solver.stringify(solver.solution))

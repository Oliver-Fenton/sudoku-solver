import os
from sudoku_image_processor import SudokuImageProcessor
from sudoku_puzzle_solver import SudokuPuzzleSolver


class Sudoku:
    """
    A wrapper class that integrates Sudoku puzzle solving and image processing.

    Combines the functionalities of SudokuImageProcessor and SudokuPuzzleSolver to provide
    a simple interface for processing, solving, and overlaying solutions onto Sudoku puzzle images.

    Attributes:
    -----------
        solver: SudokuPuzzleSolver
            An instance of SudokuPuzzleSolver to solve Sudoku puzzles.
        processor: SudokuImageProcessor
            An instance of SudokuImageProcessor to process images of Sudoku puzzles.

    Methods:
    --------
        solve_and_save(image_path, dest_path=None): Solve a Sudoku puzzle from an image and save the solution.
    """
    def __init__(self):
        self.solver = SudokuPuzzleSolver()
        self.processor = SudokuImageProcessor()

    def solve_and_save(self, image_path, dest_path=None):
        """
        Processes, solves a Sudoku puzzle from an image, and saves the solution.

        Args:
            image_path (str): Path to the image of the Sudoku puzzle.
            dest_path (str, optional): Destination path for the solved image. If None, saves in the
                                       same directory as the original image with a '_solution' suffix.

        """
        puzzle = self.processor.read_puzzle(image_path)
        self.solver.solve_puzzle(puzzle)

        if dest_path is None:
            base, extension = os.path.splitext(image_path)
            dest_path = f"{base}_solution{extension}"

        self.processor.overlay_solution(self.solver.puzzle, self.solver.solution, dest_path)

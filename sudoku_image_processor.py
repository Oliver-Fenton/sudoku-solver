import cv2
import numpy as np
import pytesseract
from imutils import grab_contours
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border


class SudokuImageProcessor:
    """
    Processes and digitizes Sudoku puzzles from images.

    This class is responsible for converting images of Sudoku puzzles into digitized
    2D arrays. It involves image processing to detect the puzzle grid, extract individual
    cells, and use OCR to recognize digits. Additionally, it can overlay the solved puzzle
    back onto the original image, creating a visual representation of the solution.

    Methods:
    --------
    read_puzzle(image_path): Recognizes and digitizes a Sudoku puzzle from an image.
    overlay_solution(puzzle, solution, dest_path): Overlays the solution onto the original image.
    """

    # Constants for the Sudoku board and OCR settings
    BOARD_SIZE = 9
    FONT = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    TESSERACT_CONFIG = '--psm 10 -c tessedit_char_whitelist=123456789'
    TESSERACT_CONFIG_2 = '--psm 6 -c tessedit_char_whitelist=123456789'

    def __init__(self, image_path=None):
        """
        Initialize the SudokuImageProcessor with an optional image path.
        Sets up initial properties and reads the puzzle if an image path is provided.
        """
        self._image_path = image_path
        self._image = None
        self._gray = None
        self._contours = None
        self._transformed_image = None
        self._puzzle = None
        if image_path:
            self.read_puzzle(image_path)

    def read_puzzle(self, image_path):
        """
        Recognize a Sudoku puzzle and return it as a 2D NumPy array.
        Loads the image, finds the puzzle contour, and extracts each cell's digit to form the Sudoku board matrix.
        """
        self._image_path = image_path
        self._image = self._load_image()
        self._gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        self._contours = self._get_puzzle_contour()
        if self._contours is None:
            raise ValueError("Failed to detect puzzle contours.")
        self._transformed_image = self._transform_image()

        # Initialize an empty board
        board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype="int")
        x = self._transformed_image.shape[1] // self.BOARD_SIZE
        y = self._transformed_image.shape[0] // self.BOARD_SIZE

        # Extract each cell's digit and fill the board
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                start_x, end_x = j * x, (j + 1) * x
                start_y, end_y = i * y, (i + 1) * y
                cell_img = self._transformed_image[start_y:end_y, start_x:end_x]
                digit = self._read_cell(cell_img)
                board[i, j] = digit

        self._puzzle = board
        return board

    def overlay_solution(self, puzzle, solution, dest_path):
        """
        Overlay the solution onto the original image and save to the given path.
        Creates a mask of the solution and applies it to the original image.
        """
        # Get mask with perspective of transformed image
        transformed_mask = self._get_transformed_solution_mask(puzzle, solution)
        # Warp the mask to the original image perspective
        mask = self._inverse_transform_image(transformed_mask)
        # Apply the mask to the original image
        self._image[mask == 255] = [0, 0, 0]
        # Save the solution image
        cv2.imwrite(dest_path, self._image)

    def _load_image(self):
        """
        Load image from the specified path.
        Raises a FileNotFoundError if the image cannot be loaded.
        """
        image = cv2.imread(self._image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or couldn't be loaded: {self._image_path}")
        return image

    @staticmethod
    def _adaptive_threshold(image):
        """
        Apply adaptive thresholding to the image.
        Uses Gaussian blurring and adaptive thresholding for better contour detection.
        """
        blurred = cv2.GaussianBlur(image, (5, 5), 3)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
        return cv2.bitwise_not(thresh)

    def _get_puzzle_contour(self):
        """
        Identify and return the contour of the sudoku puzzle.
        Applies thresholding and finds the largest 4-sided contour, assumed to be the puzzle boundary.
        """
        thresh = self._adaptive_threshold(self._gray)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = grab_contours(contours)  # get list of contours (ensures compatability with OpenCV 3/4)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Find the largest 4-sided contour
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4:
                return self._order_points(approx.reshape(4, 2)).astype(np.float32)

        return None

    @staticmethod
    def _order_points(pts):
        """
        Sort the points in clockwise order.
        This is necessary for a consistent perspective transformation.
        """
        # Initialize a list of coordinates
        rect = np.zeros((4, 2), dtype="float32")

        # The top-left point will have the smallest sum whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # The top-right point will have the smallest difference,
        # the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # Return the ordered coordinates
        return rect

    def _transform_image(self):
        """
        Transform the image to a bird's-eye view of the puzzle.
        Uses the contours to apply a perspective transformation, resulting in a top-down view.
        """
        return four_point_transform(self._gray, self._contours)

    def _inverse_transform_image(self, transformed_image):
        """
        Inverse transform the image back to its original perspective.
        Applies the inverse perspective transformation to the solution overlay.
        """
        # Define destination points for the inverse transformation
        dst = np.array([[0, 0],
                        [transformed_image.shape[1] - 1, 0],
                        [transformed_image.shape[1] - 1, transformed_image.shape[0] - 1],
                        [0, transformed_image.shape[0] - 1]],
                       dtype=np.float32)
        # Calculate the transformation matrix
        matrix = cv2.getPerspectiveTransform(dst, self._contours)
        # Apply the inverse transformation
        return cv2.warpPerspective(transformed_image, matrix, (self._image.shape[1], self._image.shape[0]))

    def _read_cell(self, cell_image):
        """
        Recognize the digit in a cell.
        """
        # Initial preprocessing with height adjustment
        processed_cell_image = self._preprocess_cell(cell_image, height_adjust=1.10)

        if processed_cell_image is None:
            return 0

        digit_str = self._ocr_digit(processed_cell_image)

        # If initial OCR fails, reprocess without height adjustment
        if not digit_str:
            processed_cell_image = self._preprocess_cell(cell_image)
            digit_str = self._ocr_digit(processed_cell_image)

        # Apply bolding at different levels if OCR still fails
        for bold_level in range(1, 4):
            if not digit_str:
                processed_cell_image = self._bold_digit(processed_cell_image, bold_level)
                digit_str = self._ocr_digit(processed_cell_image)

        # If all above fail, try thinning the digit
        if not digit_str:
            processed_cell_image = self._thin_digit(self._preprocess_cell(cell_image), 1)
            digit_str = self._ocr_digit(processed_cell_image)

        return int(digit_str[-1]) if digit_str and digit_str[-1].isdigit() else 0

    @staticmethod
    def _preprocess_cell(cell_image, height_adjust=1.0, width_adjust=1.0):
        """
        Preprocess a Sudoku cell image for OCR.
        """
        # Constants for canvas and border size
        canvas_size = 50
        border_size = 10

        # Apply thresholding to the cell image to isolate the digit
        thresh = cv2.threshold(cell_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = clear_border(thresh)

        # Find contours in the image
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = grab_contours(contours)

        # If no contours are found, return None
        if len(contours) == 0:
            return None

        # Find the largest contour, assumed to be the digit
        digit_contour = max(contours, key=cv2.contourArea)

        # Create a mask for the largest contour (digit)
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [digit_contour], -1, 255, -1)

        # Calculate the percentage of the cell filled by the digit
        (h, w) = thresh.shape
        percent_filled = cv2.countNonZero(mask) / float(w * h)
        if percent_filled < 0.03:
            return None

        # Find the bounding box of the largest contour (digit)
        x, y, w, h = cv2.boundingRect(digit_contour)
        digit = thresh[y:y + h, x:x + w]

        # Adjust the digit size based on the aspect ratio
        aspect_ratio = (float(w) * width_adjust) / (float(h) * height_adjust)
        new_height = canvas_size - (2 * border_size)
        new_width = int(aspect_ratio * new_height)
        if new_width > canvas_size - (2 * border_size):
            new_width = canvas_size - (2 * border_size)
            new_height = int(new_width / aspect_ratio)

        # If resizing results in zero width or height, return None
        if new_width == 0 or new_height == 0:
            return None

        # Resize the digit to fit the canvas
        digit_resized = cv2.resize(digit, (new_width, new_height))

        # Create a new blank canvas and place the digit in the center
        canvas = np.zeros((canvas_size, canvas_size), dtype="uint8")
        x_offset = (canvas_size - new_width) // 2
        y_offset = (canvas_size - new_height) // 2
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = digit_resized

        return canvas

    @staticmethod
    def _bold_digit(cell_image, iterations):
        """
        Make the digit in the image bolder.
        """
        # Define a 3x3 kernel for dilation
        kernel = np.ones((3, 3), np.uint8)
        # Apply dilation to the image
        return cv2.dilate(cell_image, kernel, iterations=iterations)

    @staticmethod
    def _thin_digit(cell_image, iterations):
        """
        Make the digit in the image thinner.
        """
        # Define a 3x3 kernel for erosion
        kernel = np.ones((3, 3), np.uint8)
        # Apply erosion to the image
        return cv2.erode(cell_image, kernel, iterations=iterations)

    def _ocr_digit(self, digit_image):
        """
        Recognize the digit from the image using Tesseract OCR.
        First attempt using psm 10 (treating the image as a single character).
        Second attempt using psm 6 (assuming a single uniform block of text).
        """
        # Return an empty string if the image is None
        if digit_image is None:
            return ""
        # First OCR attempt using psm 10
        digit_str = pytesseract.image_to_string(digit_image, config=self.TESSERACT_CONFIG).strip()
        # If first attempt failed, try again with psm 6
        if digit_str == "":
            digit_str = pytesseract.image_to_string(digit_image, config=self.TESSERACT_CONFIG_2).strip()
        return digit_str

    def _get_transformed_solution_mask(self, puzzle, solution):
        """
        Create a mask of the solution with the same perspective as the transformed image.
        """
        mask = np.zeros_like(self._transformed_image, dtype=np.uint8)  # Initialize a blank mask
        step_x, step_y = self._transformed_image.shape[1] // self.BOARD_SIZE, self._transformed_image.shape[0] // self.BOARD_SIZE
        avg_cell_size = (step_x + step_y) / 2.0  # Calculate the average size of a cell
        font_scale = avg_cell_size / 50.0  # Scale the font size based on cell size

        # Overlay each digit of the solution onto the mask
        for y in range(self.BOARD_SIZE):
            for x in range(self.BOARD_SIZE):
                if puzzle[y][x] != 0:  # Skip cells that already contain numbers
                    continue

                # Calculate the position for text placement
                offset_x, offset_y = x * step_x, y * step_y
                x_coord, y_coord = int(offset_x + step_x // 4), int(offset_y + 3 * step_y // 4)
                # Put the solution digit onto the mask
                cv2.putText(mask, str(solution[y][x]), (x_coord, y_coord), self.FONT, font_scale, 255, int(font_scale * 5))

        return mask

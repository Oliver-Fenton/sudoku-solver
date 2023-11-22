# Sudoku Web App

## Overview
This Flask-based web application provides a platform for users to upload images of Sudoku puzzles. The app processes these images, solves the puzzles, and displays the solutions.

## Features
- **Upload Functionality**: Users can easily upload Sudoku puzzle images.
- **Automated Puzzle Solving**: Uploaded puzzles are automatically solved by the application.
- **Solution Display and Download**: The app displays the solved puzzle and offers a download option for the solution.

## Setup and Installation
1. Clone the repository to your local machine.
2. Install the necessary dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Flask application:
```bash
python app.py
```

## Usage
- **Uploading a Puzzle**: Access the application's homepage and upload a Sudoku puzzle image.
- **Retrieving the Solution**: After processing, the app will display the solution. You have the option to download this solved puzzle image.

## Credits
Special thanks to Adrian Rosebrock for the insightful article, [OpenCV Sudoku Solver and OCR](https://pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/). The image preprocessing steps in this application were inspired and guided by the techniques covered in their article.

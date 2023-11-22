from flask import Flask, render_template, request, send_from_directory, url_for, session, redirect
import os
import shutil
from sudoku import Sudoku

# Create a Flask web application
app = Flask(__name__)

# Generate a random secret key for session management
app.config['SECRET_KEY'] = os.urandom(16).hex()

# Define the folder where uploaded Sudoku solutions will be stored
UPLOAD_FOLDER = 'static/solutions/'

# Create the UPLOAD_FOLDER directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Function to clear the contents of a directory
def clear_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


# Define the route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    # Get the session's solution filename, if available
    solution_filename = session.get('solution_filename', None)

    # Handle form submission
    if request.method == 'POST':
        file = request.files['file']
        if file:
            clear_directory(UPLOAD_FOLDER)  # Clear the solutions directory
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)

            # Solve the Sudoku puzzle and save the solution
            sudoku = Sudoku()
            sudoku.solve_and_save(filename)

            # Create a unique filename for the solution
            base, ext = os.path.splitext(file.filename)
            solution_filename = f"{base}_solution{ext}"

            # Store the solution filename in the session
            session['solution_filename'] = solution_filename

        # Redirect back to the homepage
        return redirect(url_for('index'))

    # If a solution filename is available, remove it from the session (clear solution on refresh)
    if solution_filename:
        session.pop('solution_filename', None)

    # Render the homepage HTML template and provide the solution filename
    return render_template('index.html', solution_filename=solution_filename)


# Route to serve uploaded files
@app.route('/uploads/<filename>', methods=['GET'])
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# Route to allow downloading files as attachments
@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)


# Run the Flask app if this script is executed directly
if __name__ == "__main__":
    app.run(debug=True)

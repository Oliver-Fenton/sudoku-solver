window.onload = function() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    // Drag and drop listeners
    dropZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.classList.add('dragging');
    });

    dropZone.addEventListener('dragleave', function() {
        this.classList.remove('dragging');
    });

    dropZone.addEventListener('drop', function(e) {
        e.preventDefault();
        this.classList.remove('dragging');

        const files = e.dataTransfer.files;
        fileInput.files = files;

        // Trigger form submission after a drop
        document.getElementById('uploadForm').submit();
    });

    // Click on dropZone to open the file browser
    dropZone.addEventListener('click', function() {
        fileInput.click();
    });

    // Submit the form when a file is chosen from the file browser
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            document.getElementById('uploadForm').submit();
        }
    });
};

function downloadSolution(downloadUrl) {
    window.location.href = downloadUrl;
}


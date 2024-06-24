const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const saveButton = document.getElementById('saveButton');
const clearButton = document.getElementById('clearButton');
const imageList = document.getElementById('imageList');

let isDrawing = false;

ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);
saveButton.addEventListener('click', saveImage);
clearButton.addEventListener('click', clearCanvas);

function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function draw(e) {
    if (!isDrawing) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.strokeStyle = 'black';
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
}

function saveImage() {
    const timestamp = new Date().toLocaleString().replace(/[/:]/g, '_');
    const filename = `drawing_${timestamp}.jpg`;

    // Convert canvas to JPEG data URL
    const dataURL = canvas.toDataURL('image/jpeg');

    const link = document.createElement('a');
    link.href = dataURL;
    link.download = filename;
    link.click();

    const listItem = document.createElement('li');
    const imageLink = document.createElement('a');
    imageLink.href = dataURL;
    imageLink.download = filename;
    imageLink.textContent = filename;
    listItem.appendChild(imageLink);
    imageList.appendChild(listItem);

    // If you're sending the image to the server, you might need to modify this part
    const formData = new FormData();
    formData.append('image', dataURL);

    fetch('/process_image', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.message);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}
async function uploadData() {
    const name = document.getElementById("name").value;
    const file = document.getElementById("file").files[0];
    const messageEl = document.getElementById("message");
    
    messageEl.textContent = "";
    messageEl.style.color = "black";

    if (!name || !file) {
        showError("Please enter a name and select a video file");
        return;
    }

    try {
        messageEl.textContent = "Uploading...";
        
        const formData = new FormData();
        formData.append("name", name);
        formData.append("file", file);

        console.log("Sending request to backend...");
        const response = await fetch("http://localhost:8000/upload_video", {
            method: "POST",
            body: formData,
        });

        console.log("Response received:", response);
        const data = await response.json();
        console.log("Response data:", data);
        
        if (!response.ok) {
            throw new Error(data.error || "Upload failed");
        }

        messageEl.style.color = "green";
        messageEl.textContent = data.message;

    } catch (error) {
        console.error("Upload error:", error);
        showError(error.message || "Network error");
    }
}

function showError(message) {
    const errorEl = document.getElementById("message");
    errorEl.style.color = "red";
    errorEl.textContent = message;
    console.error("Error:", message);
}

function showError(message) {
    const errorEl = document.getElementById("message");
    errorEl.style.color = "red";
    errorEl.textContent = message;
    console.error("Error:", message);
}

async function trainModel() {
    const messageEl = document.getElementById("message");
    messageEl.textContent = "";
    messageEl.style.color = "black";

    try {
        messageEl.textContent = "Training model...";
        
        const response = await fetch("http://localhost:8000/train_model", {
            method: "POST",
        });

        console.log("Response received:", response);
        const data = await response.json();
        console.log("Response data:", data);
        
        if (!response.ok) {
            throw new Error(data.error || "Training failed");
        }

        messageEl.style.color = "green";
        messageEl.textContent = data.message;

        // Redirect to recognition page after training
        setTimeout(() => {
            window.location.href = "recognition.html";
        }, 2000);  // Redirect after 2 seconds

    } catch (error) {
        messageEl.style.color = "red";
        messageEl.textContent = error.message;
        console.error("Training error:", error);
    }
}

function nextPage() {
    window.location.href = "train.html";
}

function trainModel() {
    fetch("http://localhost:8000/train_model", {
        method: "POST",
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        window.location.href = "recognition.html";
    })
    .catch(error => {
        alert("Error training model.");
    });
}

let poseModel = null;

// Initialize MediaPipe Pose
async function initializePose() {
    const pose = await window.pose.Pose.create({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
    });
    poseModel = pose;
}

// Process uploaded video
function processVideo() {
    const file = document.getElementById("videoFile").files[0];
    const video = document.getElementById("uploadedVideo");
    const canvas = document.getElementById("processingPreview");
    const ctx = canvas.getContext("2d");
    const userNameEl = document.getElementById("userName");

    if (!file) {
        userNameEl.textContent = "Please select a video!";
        userNameEl.style.color = "red";
        return;
    }

    // Load video
    video.src = URL.createObjectURL(file);
    video.play();

    // Set canvas size to match video
    video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    };

    // Process frames for pose estimation
    video.onplay = async () => {
        userNameEl.textContent = "Processing...";
        userNameEl.style.color = "white";

        // Send video to backend for prediction
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("http://localhost:8000/predict_gait", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            if (data.error) throw new Error(data.error);

            // Display user name
            userNameEl.textContent = data.user;
            userNameEl.style.color = "#00ff00";

        } catch (error) {
            userNameEl.textContent = error.message;
            userNameEl.style.color = "red";
        }
    };
}

// Initialize MediaPipe when the page loads
initializePose();

// Predict gait from uploaded video
async function predictGait(file) {
    const messageEl = document.getElementById("predictionResult");
    messageEl.textContent = "Predicting...";
    messageEl.style.color = "white";

    if (!file) {
        showError("Please select a video file");
        return;
    }

    try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch("http://localhost:8000/predict_gait", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || "Prediction failed");
        }

        messageEl.textContent = data.message;
        messageEl.style.color = "green";

    } catch (error) {
        messageEl.textContent = error.message;
        messageEl.style.color = "red";
        console.error("Prediction error:", error);
    }
}

// Show error messages
function showError(message) {
    const errorEl = document.getElementById("predictionResult");
    errorEl.textContent = message;
    errorEl.style.color = "red";
}
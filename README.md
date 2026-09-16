# EdgeVision: Real-Time Telematics & ADAS Analytics 🛡️

**Developer:** Srinjini Saha  
**Domain:** Computer Vision, Edge AI, Multithreaded Processing  

## Overview
EdgeVision is an end-to-end Advanced Driver Assistance System (ADAS) and Driver Monitoring System (DMS) engineered entirely for edge hardware. It processes dual real-time video streams to detect safety hazards and logs structured telematics data to a live web dashboard without relying on cloud computation.

## Core Architecture
This system is built on a custom multithreaded Python engine to ensure zero frame-blocking latency across simultaneous heavy AI workloads:

*   **Driver Monitoring System (DMS):** Utilizes **MediaPipe Face Mesh** to track facial landmarks and calculate Eye Aspect Ratio (EAR) for real-time fatigue and drowsiness detection.
*   **Forward Collision Warning (FCW):** Deploys **YOLOv11** (Nano) to process dashcam streams, dynamically calculating bounding box parameters to predict vehicular proximity.
*   **Thread-Safe Actuation Layer:** Features a custom offline Text-to-Speech (TTS) dispatcher using `pyttsx3` with debounce timers to bypass Windows SAPI5 deadlocks.
*   **Telematics Black Box:** Uses a local **SQLite** database to continuously log timestamped safety violations independent of the video processing threads.
*   **Live Analytics UI:** A responsive **Streamlit** web dashboard that queries the SQLite database to visualize driver safety metrics and event logs in real time.

## Technology Stack
*   **Language:** Python
*   **Computer Vision:** OpenCV, MediaPipe, YOLOv11
*   **Data & UI:** SQLite, Streamlit, Pandas
*   **System:** `threading`, `pyttsx3`

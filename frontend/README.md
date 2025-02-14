# Safe Through - Real-Time Safety and Emergency Assistance Web App

## Overview

Safe Through is a web application designed to enhance personal safety and provide real-time assistance in emergency situations. The app leverages advanced technologies like **Go Maps Pro**, **OpenLayers**, **Librosa**, and **Socket.IO** to offer features such as real-time location tracking, safe route prediction, voice sentiment analysis, and emergency response. Whether you're in distress, facing a vehicle breakdown, or simply seeking the safest route to your destination, Safe Through is here to help.

---

## Key Features

### 1. **Real-Time Location Tracking**
   - Tracks the user's real-time location using GPS.
   - Displays the location on an interactive map powered by **Go Maps Pro** and **OpenLayers**.

### 2. **Safe Route Prediction**
   - Predicts the safest route to a designated destination based on:
     - Accident-prone areas.
     - Traffic density.
     - Historical data on harmful incidents.
   - Uses a trained machine learning model to optimize route safety.

### 3. **Voice Sentiment Analysis for Emergency Detection**
   - Users can turn on the voice detection feature in case of potential danger.
   - The app uses **Librosa** to analyze voice sentiment in real-time.
   - If offensive or distressed sentiments are detected, the app automatically:
     - Triggers an **SOS alert**.
     - Notifies nearby users for assistance.

### 4. **Emergency Assistance for Vulnerable Situations**
   - Provides help in cases such as:
     - Fuel outage.
     - Vehicle breakdown.
   - Connects users with nearby assistance services or volunteers.

### 5. **Real-Time Communication with Socket.IO**
   - Enables real-time responses and notifications using **Socket.IO**.
   - Ensures seamless communication between users and emergency responders.

---

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript, OpenLayers
- **Backend**: Node.js, Express.js
- **Real-Time Communication**: Socket.IO
- **Mapping**: Go Maps Pro, OpenLayers
- **Voice Sentiment Analysis**: Librosa (Python-based)
- **Machine Learning**: Safe route prediction model (trained on accident-prone areas, traffic density, and harmful incidents)
- **Database**: MongoDB (for storing user data and incident reports)

---

## How It Works

1. **User Registration and Login**:
   - Users create an account and log in to access the app's features.

2. **Real-Time Location Tracking**:
   - The app tracks the user's location and displays it on the map.

3. **Safe Route Prediction**:
   - Users input their destination, and the app suggests the safest route based on the trained model.

4. **Voice Sentiment Analysis**:
   - In case of potential danger, users can activate the voice detection feature.
   - The app analyzes the voice sentiment and triggers an SOS alert if necessary.

5. **Emergency Assistance**:
   - Users can request help for emergencies like fuel outage or vehicle breakdown.
   - Nearby users or services are notified for assistance.

6. **Real-Time Notifications**:
   - All alerts and notifications are sent in real-time using Socket.IO.

---

## Installation and Setup

### Prerequisites
- Node.js and npm installed.
- Python installed (for Librosa and sentiment analysis).
- MongoDB installed and running.

### Steps to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone
   cd safe-through
   ```

2. **Install Dependencies**:
   ```bash
   npm install
   ```

3. **Set Up Environment Variables**:
   - Create a `.env` file in the root directory and add the following:
     ```
     MONGO_URI=your_mongodb_connection_string
     SOCKET_IO_SECRET=your_socket_io_secret
     GOOGLE_MAPS_API_KEY=your_google_maps_api_key
     ```

4. **Run the Backend Server**:
   ```bash
   npm start
   ```

5. **Run the Voice Sentiment Analysis Script**:
   - Navigate to the `voice-analysis` directory and run:
     ```bash
     python sentimentAnalysis.py
     ```

6. **Access the Web App**:
   - Open your browser and go to `http://localhost:5173`.

---

## Future Enhancements

- Integration with local emergency services.
- AI-based predictive analytics for crime-prone areas.
- Mobile app development for better accessibility.
- Multi-language support for voice sentiment analysis.

---

## Contributing

We welcome contributions! If you'd like to contribute to Safe Through, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Submit a pull request.
---

**Stay Safe with Safe Through!**

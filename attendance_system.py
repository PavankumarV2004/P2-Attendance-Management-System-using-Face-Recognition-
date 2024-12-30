import streamlit as st
import cv2
import os
import sqlite3
from datetime import datetime
import numpy as np

def setup_database():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS students (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        enrollment TEXT UNIQUE,
                        name TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        enrollment TEXT,
                        date TEXT,
                        time TEXT)''')
    conn.commit()
    conn.close()

def add_student(enrollment, name):
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO students (enrollment, name) VALUES (?, ?)", (enrollment, name))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def remove_student(enrollment):
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("DELETE FROM students WHERE enrollment=?", (enrollment,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

def capture_images(enrollment, name):
    if not os.path.exists('TrainingImages'):
        os.makedirs('TrainingImages')

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    sampleNum = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sampleNum += 1
            cv2.imwrite(f"TrainingImages/{name}.{enrollment}.{sampleNum}.jpg", gray[y:y + h, x:x + w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Capturing Images', img)

        # Stop capturing after 30 images
        if sampleNum >= 30:
            break

    cam.release()
    cv2.destroyAllWindows()

def train_images():
    recognizer = cv2.face_LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = []
    ids = []

    for imagePath in [os.path.join('TrainingImages', f) for f in os.listdir('TrainingImages')]:
        img = cv2.imread(imagePath, 0)
        id_ = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(img)
        ids.append(id_)

    recognizer.train(faces, np.array(ids))
    if not os.path.exists('TrainedModel'):
        os.makedirs('TrainedModel')
    recognizer.save('TrainedModel/trainer.yml')

def mark_attendance():
    recognizer = cv2.face_LBPHFaceRecognizer_create()
    recognizer.read('TrainedModel/trainer.yml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0)
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            id_, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 50:
                cursor.execute("SELECT name FROM students WHERE id=?", (id_,))
                student = cursor.fetchone()
                if student:
                    name = student[0]
                    now = datetime.now()
                    date = now.strftime('%Y-%m-%d')
                    time = now.strftime('%H:%M:%S')

                    cursor.execute("INSERT INTO attendance (enrollment, date, time) VALUES (?, ?, ?)", (id_, date, time))
                    conn.commit()

                    cv2.putText(img, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    cv2.putText(img, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else:
                cv2.putText(img, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Mark Attendance', img)

        # Stop the camera when Escape key (27) is pressed
        if cv2.waitKey(1) & 0xFF == 27:  # 27 is the Escape key
            break

    cam.release()
    cv2.destroyAllWindows()
    conn.close()

def manual_attendance(enrollment, date, time):
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO attendance (enrollment, date, time) VALUES (?, ?, ?)", (enrollment, date, time))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

setup_database()
st.title("Attendance Management System")

menu = ["Home", "Register Student", "Remove Student", "Train Model", "Mark Attendance", "Manual Attendance"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("Welcome to the Attendance Management System")
    st.write("Use the sidebar to navigate.")

elif choice == "Register Student":
    st.subheader("Register a New Student")
    enrollment = st.text_input("Enrollment ID")
    name = st.text_input("Student Name")
    if st.button("Register"):
        if enrollment and name:
            if add_student(enrollment, name):
                st.success(f"Student {name} registered successfully!")
                capture_images(enrollment, name)
                st.info("Images captured successfully!")
            else:
                st.warning("Enrollment ID already exists!")
        else:
            st.error("Please fill out all fields.")

elif choice == "Remove Student":
    st.subheader("Remove a Student")
    enrollment = st.text_input("Enter Enrollment ID to remove")
    if st.button("Remove Student"):
        if enrollment:
            if remove_student(enrollment):
                st.success(f"Student with Enrollment ID {enrollment} removed successfully!")
            else:
                st.error(f"Student with Enrollment ID {enrollment} not found.")
        else:
            st.error("Please enter the Enrollment ID.")

elif choice == "Train Model":
    st.subheader("Train the Model")
    if st.button("Train"):
        train_images()
        st.success("Model trained successfully!")

elif choice == "Mark Attendance":
    st.subheader("Mark Attendance")
    if st.button("Start Attendance"):
        mark_attendance()
        st.success("Attendance marked successfully!")

elif choice == "Manual Attendance":
    st.subheader("Manually Mark Attendance")
    enrollment = st.text_input("Enrollment ID")
    date = st.date_input("Date")
    time = st.time_input("Time")
    if st.button("Submit Attendance"):
        if enrollment:
            if manual_attendance(enrollment, date.strftime('%Y-%m-%d'), time.strftime('%H:%M:%S')):
                st.success("Attendance marked manually successfully!")
            else:
                st.error("Failed to mark attendance. Check details.")
        else:
            st.error("Please enter enrollment ID.")

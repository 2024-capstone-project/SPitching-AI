import streamlit as st
import pandas as pd
import os
import cv2
import sqlite3
import datetime
import plotly.express as px
from displayDB import display_tables_and_contents

# Get the current date and time
current_datetime = datetime.datetime.now()

# Format it as a string including seconds and with the desired format
current_date = current_datetime.strftime("%dth %b %H:%M:%S")

# Create or connect to the SQLite database
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create a table to store user data if it doesn't exist
c.execute('''
          CREATE TABLE IF NOT EXISTS users (
              username TEXT PRIMARY KEY,
              password TEXT
          )
          ''')
conn.commit()

def create_user_scores_table(username):
    try:
        c.execute(f'''
            CREATE TABLE IF NOT EXISTS {username}_scores (
                username TEXT,  
                head_score INTEGER,
                eye_score INTEGER,
                smile_score INTEGER,
                hand_score INTEGER,
                pose_score INTEGER,
                date_added DATE,
                avg_score INTEGER,
                FOREIGN KEY (username) REFERENCES users(username)
            )
        ''')
        conn.commit()
        #st.success(f"Table created successfully for user: {username}")
    except Exception as e:
        st.error(f"Error creating account for user {username}: {e}")

# Function to update user scores in the database
def update_user_scores(username, head_score, eye_score, smile_score, hand_score, pose_score, date, avg_score):
    c.execute(f'''
              INSERT INTO {username}_scores
              (username, head_score, eye_score, smile_score, hand_score, pose_score, date_added, avg_score)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)
              ''', (username, head_score, eye_score, smile_score, hand_score, pose_score, date, avg_score))
    conn.commit()

# Set up Streamlit session state
if 'state' not in st.session_state:
    st.session_state.state = 'login'

# Main Streamlit app
if st.session_state.state == 'login':
    st.title("Login")
    login_username = st.text_input("Username:")
    login_password = st.text_input("Password:", type="password")

    if st.button("Login"):
        # Check if username and password are not blank
        if not login_username or not login_password:
            st.error("Username and password cannot be blank.")
        else:
            # Check username and password (replace with your authentication logic)
            users = c.execute('''
                                 SELECT * FROM users
                                 WHERE username = ? AND password = ?
                                 ''', (login_username, login_password)).fetchone()

            if users:
                st.success("Login successful!")
                st.session_state.username = login_username  # Store username in session state
                st.session_state.state = 'main'
            else:
                st.error("Incorrect username or password. Please try again.")

    st.markdown("---")
    st.subheader("Don't have an account?")
    if st.button("Sign Up"):
        st.session_state.state = 'signup'

elif st.session_state.state == 'signup':
    st.title("Sign Up")
    signup_username = st.text_input("New Username:")
    signup_password = st.text_input("New Password:", type="password")
    confirm_password = st.text_input("Confirm Password:", type="password")

    if st.button("Sign Up"):
        # Check if username, password, and confirm password are not blank
        if not signup_username or not signup_password or not confirm_password:
            st.error("Username, password, and confirm password cannot be blank.")
        else:
            if signup_password == confirm_password:
                existing_user = c.execute('''
                                          SELECT * FROM users
                                          WHERE username = ?
                                          ''', (signup_username,)).fetchone()

                if existing_user:
                    st.error("Username already taken. Please choose a different one.")
                else:
                    c.execute('''
                              INSERT INTO users (username, password)
                              VALUES (?, ?)
                              ''', (signup_username, signup_password))
                    conn.commit()
                    
                    # Create a table for the new user's scores
                    create_user_scores_table(signup_username)
                    #create_user_scores_table(signup_username)

                    st.success("Account created successfully! You can now log in.")
                    st.session_state.state = 'login'
            else:
                st.error("Passwords do not match. Please try again.")

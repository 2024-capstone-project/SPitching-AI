import streamlit as st

def main():
    st.title("Simple Greeting App")
    
    # Get user input
    name = st.text_input("Enter your name:")
    color_options = ['Red', 'Blue', 'Green', 'Yellow']
    favorite_color = st.selectbox("Select your favorite color:", color_options)
    
    # Display greeting message
    if name:
        st.write(f"Hello, {name}! Your favorite color is {favorite_color}.")

if __name__ == "__main__":
    main()

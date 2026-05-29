import streamlit as st

from fileingestorV2 import FileIngestorV2
from streamlit_extras.app_logo import add_logo
from streamlit_extras.colored_header import colored_header

# Set the title for the Streamlit app
#st.title("Chat with Pixii Bot 🤗")

#st.title(":black[Chat with iDOC 🤗]")
st.title(":green[Brikshyan iDOC chat ]")

colored_header(label='', description='', color_name='gray-30')

#logo_link = "iDOC_logo.jpg"
logo_link = "Brikshyan.jpg"
st.sidebar.image(logo_link)
# Create a file uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload File", type="pdf")

if uploaded_file:
    file_ingestor = FileIngestorV2(uploaded_file)
    file_ingestor.handlefileandingest()
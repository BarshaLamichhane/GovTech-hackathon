# import streamlit as st

# from fileingestorGovTech import FileIngestorGovTech
# from streamlit_extras.app_logo import add_logo
# from streamlit_extras.colored_header import colored_header

# # Set the title for the Streamlit app
# #st.title("Chat with Pixii Bot 🤗")

# #st.title(":black[Chat with iDOC 🤗]")
# st.title(":blue[AI Knowledge Bot for GovTech reports ]")

# colored_header(label='', description='', color_name='gray-30')

# #logo_link = "iDOC_logo.jpg"
# logo_link = "Hackathon ENG.png"
# st.sidebar.image(logo_link)
# # Create a file uploader in the sidebar
# #####single pdf file uploader
# # uploaded_file = st.sidebar.file_uploader("Upload File", type="pdf")

# # if uploaded_file:
# #     file_ingestor = FileIngestorGovTech(uploaded_file)
# #     file_ingestor.handlefileandingest()
# ############################

# #########################multiple pdf file uploader
# # uploaded_files = st.sidebar.file_uploader(
# #     "Upload PDF reports",
# #     type="pdf",
# #     accept_multiple_files=True
# # )

# # if uploaded_files:
# #     file_ingestor = FileIngestorGovTech(uploaded_files)
# #     file_ingestor.handlefileandingest()
# ##########################

# uploaded_files = st.sidebar.file_uploader(
#     "Upload PDF reports",
#     type="pdf",
#     accept_multiple_files=True
# )

# if uploaded_files:
#     file_ingestor = FileIngestorGovTech(uploaded_files)
#     file_ingestor.handlefileandingest()



import streamlit as st

from fileingestorGovTech import FileIngestorGovTech
from streamlit_extras.colored_header import colored_header

st.title(":blue[AI Knowledge Bot for GovTech Reports]")

colored_header(label="", description="", color_name="gray-30")

logo_link = "Hackathon ENG.png"
st.sidebar.image(logo_link)

mode = st.sidebar.radio(
    "Choose mode",
    [
        "Chat with reports",
        "AI Project Readiness Advisor",
        "Risk Dashboard"
    ]
)

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF reports (optional)",
    type="pdf",
    accept_multiple_files=True
)

file_ingestor = FileIngestorGovTech(uploaded_files, mode)
file_ingestor.handlefileandingest()

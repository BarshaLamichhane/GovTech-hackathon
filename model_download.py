import requests

# URL of the model file
url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q2_K.bin"
url_2= "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"

# Path where the model will be saved
save_path = './models/llama-2-7b-chat.ggmlv3.q2_K.bin'
### Added later by Barsha starts here ########
save_path_url_2 = './models/llama-2-7b-chat.Q4_K_M.gguf'
### Added later by Barsha ends here ########
# Download the model file
response = requests.get(url, stream=True)

# Check if the download was successful
if response.status_code == 200:
    # Open the file in write-binary mode and save the content
    # with open(save_path, 'wb') as f:
    #     for chunk in response.iter_content(chunk_size=1024):
    #         if chunk:
    #             f.write(chunk)
    with open(save_path_url_2, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


    print(f'Model downloaded successfully and saved to {save_path_url_2}')
else:
    print('Error downloading the model.')

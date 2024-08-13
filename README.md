# Medical-Chatbot

Clone Git Project repo: https://github.com/TIFFMUN/Medical-Chatbot-2.git

Create Virtual Environment
conda create -n mchatbot python=3.8 -y

Activate Virtual Environment
conda activate mchatbot

Go into Folder
dir 
cd Medical-Chatbot-2

Install Requirements 
python -m pip install -r requirements.txt

Create .env file in root directory and add Pinecone credentials 

Download the Llama 2 Model:
llama-2-7b-chat.ggmlv3.q4_0.bin
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

Load store_index.py file 
python store_index.py

Load app.py file 
python app.py 

Open up Local Host to Run Medical Chatbot 

Ctrl + C to stop Medical Chatbot

Techstacks Used: 
- Python 
- LangChain
- Flask 
- Meta Llama 2
- Pinecone 


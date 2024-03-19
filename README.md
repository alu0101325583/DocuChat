# DocuChat
Python program to generate a vector db over a set of documents and enable llm chatting about them

## Requirements
- Python >= 3.8 && < 4

- OpenAi API key

## Install:

### Configuration:

Create a file called ".env" and inside, place you api key just like this
```bash
OPENAI_API_KEY=your_api_key_goes_here
```

Create a virtual Environment:
```bash
pip3 install virtualenv
python3 -m venv chatenv
```

Activate it:
```bash
# Mac/Linux
source chatenv/bin/activate

# Windows cmd
chatenv\Scripts\activate.bat
```

### Install dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```


## Usage 

- Execution; Be sure to have the environment active and then run the following command:
```bash
  python main.py -d path/to/documents/folder
```

This will check if that folder contains an existing database, otherwise it will process all the documents inside and generate a vector database in that path

After having loaded the database, a chat interface will appear to let you ask a LLM about those documents.

- Exit: To exit just type exit() inside the chat interface

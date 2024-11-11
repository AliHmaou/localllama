# localllama
Minimalistic local llama projet for hardware ability tests (Apple M4)

## Setup
1. Clone this repo
`git clone https://github.com/alihmaou/localllama`
2. Optional : set up a virtual environement in the downloaded folder (example for venv in a folder called .venv, on windows) then activate it
`python -m venv .venv`
`source .venv/bin/activate` (Windows : .venv\Scripts\activate)
3. Load and install the requirements (llama-cpp-python takes up to 1 hour on windows)
`pip install -r requirements.txt`
4. Run the app.py file 
`python app.py`
5. Access the gradio app on https://127.0.0.1:7860 (change share=False to Share=True to obtain a public url through gradio tunnel)
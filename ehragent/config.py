import os
from dotenv import load_dotenv

load_dotenv()

def openai_config(model):
    config = {
        "model": model,
        "api_key": os.getenv("OPENAI_API_KEY")
    }    
    return config

def llm_config_list(seed, config_list):
    llm_config_list = {
        "functions": [
            {
                "name": "python",
                "description": "run the entire code and return the execution result. Only generate the code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell": {
                            "type": "string",
                            "description": "Valid Python code to execute.",
                        }
                    },
                    "required": ["cell"],
                },
            },
        ],
        "config_list": config_list,
        "timeout": 120,
        "cache_seed": seed,
        #"temperature": 0,
    }
    return llm_config_list
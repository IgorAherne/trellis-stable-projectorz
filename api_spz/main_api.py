import os
import sys
import platform
import torch

# -------------LOW VRAM TESTING -------------


# # only used for debugging, to emulate low-vram graphics cards:
#
#torch.cuda.set_per_process_memory_fraction(0.43)  # Limit to 43% of my available VRAM, for testing.
# And/or set maximum split size (in MB)
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.8'


# -------------- INFO LOGGING ----------------

print(
    f""
    f"[System Info] Python: {platform.python_version():<8} | "
    f"PyTorch: {torch.__version__:<8} | "
    f"CUDA: {'not available' if not torch.cuda.is_available() else torch.version.cuda}"
)

import logging

class TritonFilter(logging.Filter):# Custom filter to ignore Triton messages
    def filter(self, record):
        message = record.getMessage().lower()
        triton_phrases = [
            "triton is not available",
            "matching triton is not available",
            "no module named 'triton'"
        ]
        return not any(phrase in message for phrase in triton_phrases)
    
logger = logging.getLogger("trellis")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent messages from propagating to root
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',  datefmt='%H:%M:%S'))
#and to our own "trellis" logger:
handler.addFilter(TritonFilter())
logger.addHandler(handler)
# other scripts can now use this logger by doing 'logger = logging.getLogger("trellis")'

# Configure root logger:
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S', # Only show time
    handlers=[logging.StreamHandler()]
)
# Apply TritonFilter to all root handlers
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.addFilter(TritonFilter())



# -------------- CMD ARGS PARSE -----------------

# read command-line arguments, passed into this script when launching it:
import argparse
parser = argparse.ArgumentParser(description="Trellis API server")

parser.add_argument("--precision", 
                    choices=["full", "half", "float32", "float16"], 
                    default="full",
                    help="Set the size of variables for pipeline, to save VRAM and gain performance")

parser.add_argument("--ip", 
                    type=str, 
                    default="127.0.0.1", 
                    help="Specify the IP address on which the server will listen (default: 127.0.0.1)")

parser.add_argument("--port", 
                    type=int, 
                    default=7960, 
                    help="Specify the port on which the server will listen (default: 7960)")

cmd_args = parser.parse_args()


# -------------- PIPELINE SETUP ----------------

var_cwd = os.getcwd()
sys.path.append(var_cwd)

print('')
logger.info("Trellis API Server is starting up:")
logger.info("Touching this window will pause it.  If it happens, click inside it and press 'Enter' to unpause")
print('')

# Configure environment, BEFORE including trellis pipeline
os.environ['ATTN_BACKEND'] = 'xformers'    # or 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'       # or 'auto'

# IMPORTING FROM state_manage AND INITIALIZE THE TRELLIS PIPELINE,
# So, importing it only AFTER all of the above setup:
from api_spz.core.state_manage import state 
state.initialize_pipeline(cmd_args.precision)


# -------------- API SERVER SETUP AND LAUNCH ----------------

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_spz.routes import generation
from version import code_version

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('')
    logger.info(f"Trellis API version {code_version}")
    logger.info(f"Trellis API Server is active and listening on {cmd_args.ip}:{cmd_args.port}")
    print('')
    yield
    state.cleanup()#shutdown

app = FastAPI(title="Trellis API", lifespan=lifespan)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add the generation router
app.include_router(generation.router)

if __name__ == "__main__":
    uvicorn.run( app,  
                 host=cmd_args.ip,
                 port=cmd_args.port,  
                 log_level="warning" )
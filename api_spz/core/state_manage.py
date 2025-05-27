import torch
from pathlib import Path
from trellis.pipelines import TrellisImageTo3DPipeline

class TrellisState:
    def __init__(self):
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)

    def cleanup(self):
        if self.temp_dir.exists():
            from shutil import rmtree
            rmtree(self.temp_dir)

    # Define a function to initialize the pipeline
    def initialize_pipeline(self, precision="full"):
        global pipeline
        pipeline = TrellisImageTo3DPipeline.from_pretrained("jetx/TRELLIS-image-large")
        # Apply precision settings. Reduce memory usage at the cost of numerical precision:
        print('')
        print(f"used precision: '{precision}'.  Loading...")
        if precision == "half" or precision=="float16":
            pipeline.to(torch.float16) #cuts memory usage in half
            if "image_cond_model" in pipeline.models:
                pipeline.models['image_cond_model'].half()  #cuts memory usage in half
        # Attach the pipeline to the state object:
        state.pipeline = pipeline
        # DO NOT MOVE TO CUDA YET. We'll be dynamically loading parts between 'cpu' and 'cuda' soon.
        # Kept for precaution:
        #    pipeline.cuda()

# Global state instance:
state = TrellisState()
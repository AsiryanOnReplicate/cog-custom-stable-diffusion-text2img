# Prediction interface for Cog ⚙️
from cog import BasePredictor, Input, Path
import os
import math
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image
from diffusers import (DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler
)

MODEL_CACHE = "cache"

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_CACHE, safety_checker = None, 
            custom_pipeline="lpw_stable_diffusion", 
            revision="fp16", 
            torch_dtype=torch.float16
        )
        self.pipe = pipe.to("cuda")
        
    def base(self, x):
        return int(8 * math.floor(int(x)/8))

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="RAW photo, a portrait photo of a latina woman in casual clothes, natural skin, 8k uhd, high quality, film grain, Fujifilm XT3",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        ),
        width: int = Input(description="Width", ge=0, le=1920, default=512),
        height: int = Input(description="Height", ge=0, le=1920, default=728),
        strength: float = Input(description="Strength/weight", ge=0, le=1, default=1.0),
        num_inference_steps: int = Input(description="Number of inference steps", ge=0, le=100, default=20),
        guidance_scale: float = Input(
            description="Guidance scale", ge=0, le=10, default=7.5
        ),
        scheduler: str = Input(
            description="Scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER_ANCESTRAL",
        ),
        use_karras_sigmas: bool = Input(description="Use karras sigmas or not", default=False),
        seed: int = Input(description="Leave blank to randomize", default=None),
    ) -> Path:
        """Run a single prediction on the model"""
        if (seed == 0) or (seed == None):
            seed = int.from_bytes(os.urandom(2), byteorder='big')
        generator = torch.Generator('cuda').manual_seed(seed)
        self.pipe.scheduler = SCHEDULERS[scheduler].from_config(self.pipe.scheduler.config, use_karras_sigmas=use_karras_sigmas)
        
        print("Scheduler:", scheduler)
        print("Using karras sigmas:", use_karras_sigmas)
        print("Using seed:", seed)

        output_image = self.pipe(
            prompt=prompt,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=self.base(width),
            height=self.base(height),
            negative_prompt=negative_prompt,
            generator=generator,
        ).images[0]
        
        out_path = Path(f"/tmp/output.png")
        output_image.save(out_path)
        return  out_path

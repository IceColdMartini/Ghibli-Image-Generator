# generate_ghibli_with_ipadapter.py

from ip_adapter import IPAdapter
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Load base model
model_id = "runwayml/stable-diffusion-v1-5"

# Load pipeline with MPS acceleration (Apple Silicon GPU)
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True
).to("mps")

# Initialize IP-Adapter
ip_model = IPAdapter(
    pipe,
    image_encoder_path="models/image_encoder",
    ip_ckpt="models/ip-adapter_sd15.bin",
    device="mps"
)

# Load client's photos
person_a = Image.open("client_photos/person_a.jpg").convert("RGB")
person_b = Image.open("client_photos/person_b.jpg").convert("RGB")

# Prompt for the scene
prompt = (
    "Two people sitting under a Christmas tree exchanging gifts, "
    "anime style, Studio Ghibli, soft lighting, cinematic composition, "
    "hand drawn, watercolor, warm colors"
)

negative_prompt = "low quality, blurry, deformed, ugly, bad proportions"

# Generate image with both faces
image = ip_model.generate(
    prompt=prompt,
    negative_prompt=negative_prompt,
    images=[person_a, person_b],
    num_samples=1,
    num_inference_steps=50,
    guidance_scale=7.5,
    cross_attention_kwargs={"scale": 0.8}
)[0]

# Save output
output_path = "./outputs/ghibli_scene_with_faces.png"
image.save(output_path)
print(f"✅ Image saved at: {output_path}")
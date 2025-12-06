import os
import torch
from typing import List, Tuple, Optional, Union
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor, PreTrainedTokenizer

# Import custom modules from the MS-Diffusion implementation
from msdiffusion.models.projection import Resampler
from msdiffusion.models.model import MSAdapter
from msdiffusion.utils import get_phrase_idx, get_eot_idx


class MSDiffusionGenerator:
    def __init__(
        self,
        base_model_path: str,
        image_encoder_path: str,
        adapter_ckpt_path: str,
        device: str = None
    ):
        """
        Initializes the MS-Diffusion generation pipeline.

        Args:
            base_model_path (str): Path to the Stable Diffusion XL base model.
            image_encoder_path (str): Path to the CLIP image encoder.
            adapter_ckpt_path (str): Path to the MS-Adapter checkpoint (.bin file).
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing MSDiffusionGenerator on {self.device}...")

        # 1. Load Base SDXL Pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            add_watermarker=False,
        ).to(self.device)

        # 2. Load CLIP Image Encoder
        self.image_encoder_type = "clip"
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path
        ).to(self.device, dtype=torch.float16)
        
        # 3. Initialize Image Processor
        self.image_processor = CLIPImageProcessor()

        # 4. Build Multimodal Fusion Modules
        self.num_tokens = 16
        self.image_proj_type = "resampler"
        
        # Resampler: Maps image features to the text latent space
        self.image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
            latent_init_mode="grounding",
            phrase_embeddings_dim=self.pipe.text_encoder.config.projection_dim,
        ).to(self.device, dtype=torch.float16)

        # MSAdapter: Injects the projection model into the UNet
        self.ms_model = MSAdapter(
            self.pipe.unet,
            self.image_proj_model,
            ckpt_path=adapter_ckpt_path,
            device=self.device,
            num_tokens=self.num_tokens
        ).to(self.device, dtype=torch.float16)
        
        print("Models loaded successfully.")

    @staticmethod
    def get_phrases_idx(
        tokenizer: PreTrainedTokenizer, 
        phrases: List[str], 
        prompt: str
    ) -> List[int]:
        """
        Finds the token indices for specific phrases within a tokenized prompt.

        Args:
            tokenizer: The tokenizer used by the model.
            phrases (List[str]): List of phrases to locate (e.g., ['dog', 'cat']).
            prompt (str): The complete prompt string.

        Returns:
            List[int]: A list of token indices corresponding to the phrases.
        """
        res = []
        phrase_cnt = {}
        
        for phrase in phrases:
            # Track occurrences to handle duplicate words correctly
            if phrase in phrase_cnt:
                phrase_cnt[phrase] += 1
            else:
                phrase_cnt[phrase] = 1
                
            # Use utility to get index (returns a tuple, we take the first element)
            # Note: get_phrase_idx is imported from msdiffusion.utils
            idx = get_phrase_idx(tokenizer, phrase, prompt, num=phrase_cnt[phrase]-1)[0]
            res.append(idx)
            
        return res

    def generate(
        self,
        image_path: str,
        prompt: str,
        phrases: List[str],
        save_dir: str,
        save_name: str,
        num_samples: int = 5,
        steps: int = 30
    ) -> List[str]:
        """
        Generates poster images based on an input image and text constraints.

        Args:
            image_path (str): Path to the reference product image.
            prompt (str): Description of the poster (e.g., "A shoe in a playground").
            phrases (List[str]): Keywords in the prompt that map to the image (e.g., ["shoes"]).
            save_dir (str): Directory to save outputs.
            save_name (str): Base filename for saved images.
            num_samples (int): Number of images to generate.
            steps (int): Inference steps.

        Returns:
            List[str]: Paths to the saved images.
        """
        if not os.path.exists(image_path):
            print(f"Error: Input image not found at {image_path}")
            return []

        # 1. Preprocess Input Image
        try:
            raw_image = Image.open(image_path)
            input_image = raw_image.convert("RGB").resize((512, 512))
        except Exception as e:
            print(f"Error loading image: {e}")
            return []

        print(f"Generating with prompt: '{prompt}'")
        
        # 2. Prepare Constraints
        # Define bounding box (Center crop assumption: [y1, x1, y2, x2])
        boxes = [[[0.25, 0.25, 0.75, 0.75]]] 
        drop_grounding_tokens = [0] # 0 = Keep tokens, 1 = Drop

        # Calculate attention indices
        # Note: phrases input to generate requires a list of lists for batch processing
        # phrases param here expects: [['keyword']]
        batch_phrases = [phrases] 
        
        phrase_idxes = [self.get_phrases_idx(self.pipe.tokenizer, phrases, prompt)]
        
        # EOT (End of Text) indices used for masking or attention bounds
        eot_idxes = [[get_eot_idx(self.pipe.tokenizer, prompt)] * len(phrases)]

        print(f"Phrase Indices: {phrase_idxes}")
        print(f"EOT Indices: {eot_idxes}")

        # 3. Run Inference
        generated_images = self.ms_model.generate(
            pipe=self.pipe,
            pil_images=[[input_image]], # Double bracket for batch dim
            num_samples=num_samples,
            num_inference_steps=steps,
            seed=0,
            prompt=[prompt],
            scale=0.6,
            image_encoder=self.image_encoder,
            image_processor=self.image_processor,
            boxes=boxes,
            image_proj_type=self.image_proj_type,
            image_encoder_type=self.image_encoder_type,
            phrases=batch_phrases,
            drop_grounding_tokens=drop_grounding_tokens,
            phrase_idxes=phrase_idxes,
            eot_idxes=eot_idxes,
            height=1024,
            width=1024
        )

        # 4. Save Results
        output_paths = []
        full_save_path = os.path.join(save_dir, save_name)
        os.makedirs(full_save_path, exist_ok=True)

        for i, img in enumerate(generated_images):
            file_path = os.path.join(full_save_path, f"{i}.jpg")
            img.save(file_path)
            output_paths.append(file_path)
            print(f"Saved image to: {file_path}")

        return output_paths

def get_subject_from_image_name(image_path: str) -> str:
    """
    Extracts a potential subject keyword from the image filename.
    Example: 'path/to/Red_Sneaker.jpg' -> 'red_sneaker'
    """
    filename = os.path.basename(image_path)
    name_no_ext = os.path.splitext(filename)[0]
    clean_name = name_no_ext.lower().replace(" ", "_")
    return clean_name

if __name__ == '__main__':
    # --- Configuration ---
    # Ideally, use Environment Variables or a Config file for these paths
    BASE_MODEL_PATH = "/home/rjiangas/cv_project/MS-Diffusion/models/tabilityai/stable-diffusion-xl-base-1.0"
    IMAGE_ENCODER_PATH = "/home/rjiangas/cv_project/MS-Diffusion/models/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    ADAPTER_CKPT_PATH = "/home/rjiangas/cv_project/MS-Diffusion/models/doge1516/MS-Diffusion/ms_adapter.bin"
    
    RESULT_DIR = "./res"
    
    # --- Input Data ---
    image_file = 'example/snaker.jpeg'
    # image_file = 'example/toy.jpg'
    
    # Ensure the phrase exists in the prompt!
    phrase_keyword = "shoe"
    # phrase_keyword = get_subject_from_image_name(image_file) # e.g., "snaker"
    # phrase_keyword = "toy"
    
    # poster_prompt = (
    #     "\"Better is Temporary\" in English, Kinetic Brush style, minimalist dark background, dynamic typography with flying white effect, glowing energy trail, monochrome with electric blue accents, powerful and aspirational mood, masterpiece, best quality, 8k. A single shoe anchors the composition."
    # )
    poster_prompt = (
         "slogan=\"Innovation Challenges Tradition\", Mecha Sci-Fi style, dark futuristic cityscape, the shoe integrated into deconstructed typography with sharp angles, neon accents and metallic textures, metallic blues and silvers with vibrant magenta glow, edgy and progressive mood, masterpiece, best quality, 8k." 
    )   
    # poster_prompt = (
    #     "slogan=\"Let's go cause trouble together\", Urban Graffiti style, textured brick wall background, bold, layered typography with spray paint drips, a toy sits prominently in the foreground, vibrant neon and primary colors, rebellious and energetic mood, masterpiece, best quality, 8k."
    # )   
    # --- Execution ---
    try:
        generator = MSDiffusionGenerator(
            base_model_path=BASE_MODEL_PATH,
            image_encoder_path=IMAGE_ENCODER_PATH,
            adapter_ckpt_path=ADAPTER_CKPT_PATH
        )

        saved_files = generator.generate(
            image_path=image_file,
            prompt=poster_prompt,
            phrases=[phrase_keyword], # Must be a list of strings
            save_dir=RESULT_DIR,
            save_name=phrase_keyword,
            num_samples=5
        )
        
    except Exception as e:
        print(f"Critical failure in main execution: {e}")
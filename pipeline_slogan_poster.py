import os
import torch
import gc
import argparse
from typing import Dict, Any, List

# --- 1. Import Custom Modules ---
from utils.slogan_gen import MarketingContentGenerator
from utils.poster_prompt_gen import PosterPromptGenerator
from utils.poster_gen import MSDiffusionGenerator


class AutoPosterPipeline:
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Pipeline configuration.
        Note: Models are not loaded immediately to save VRAM. 
        They will be loaded and unloaded dynamically during the run method.
        """
        self.config = config
        
    def _clean_memory(self):
        """
        Force cleans GPU cache and System RAM to prevent Out-Of-Memory (OOM) errors.
        This is crucial when running multiple large models (VLM + SDXL) sequentially.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        print(">> [System] Memory Cleaned.")

    def get_phrase_from_filename(self, path: str) -> str:
        """
        Extracts the product keyword from the filename.
        Example: 'example/my_sneaker.jpeg' -> 'my_sneaker'
        """
        name = os.path.splitext(os.path.basename(path))[0]
        # Normalize: lowercase and replace spaces with underscores
        return name.lower().replace(" ", "_")

    def run(self, image_path: str, user_requirements: Dict[str, Any], save_dir: str):
        """
        Executes the full generative workflow:
        1. VLM: Image -> Creative Slogan
        2. LLM: Slogan + Keywords -> Visual Poster Prompt
        3. MS-Diffusion: Visual Prompt + Image -> Final Poster
        """
        
        # 0. Preparation
        if not os.path.exists(image_path):
            print(f"Error: Input image not found at {image_path}")
            return

        product_phrase = self.get_phrase_from_filename(image_path)
        print(f"\n{'='*20} Pipeline Start: {product_phrase} {'='*20}")

        # ---------------------------------------------------------
        # STEP 1: VLM - Generate Slogan
        # ---------------------------------------------------------
        print("\n>> [Phase 1] Analyzing Image & Generating Slogan...")
        
        # 1.1 Load VLM
        vlm = MarketingContentGenerator(
            model_path=self.config["vlm_model_path"]
        )
        
        # 1.2 Generate Content
        marketing_content = vlm.generate(
            image_path=image_path,
            **user_requirements  # Unpack brand_tone, platform, etc.
        )
        
        # 1.3 Validate and Select
        if not marketing_content or not marketing_content.get("slogans"):
            print("!! [Error] VLM failed to generate slogans. Aborting.")
            return
        
        # Strategy: Select the first generated slogan
        selected_slogan = marketing_content["slogans"][0]
        print(f"   Selected Slogan: \"{selected_slogan}\"")
        
        # 1.4 Critical: Unload VLM to free VRAM for SDXL
        del vlm
        self._clean_memory()


        # ---------------------------------------------------------
        # STEP 2: LLM - Generate Visual Prompt
        # ---------------------------------------------------------
        print("\n>> [Phase 2] Engineering Visual Prompt via LLM...")
        
        # 2.1 Initialize LLM Client (Lightweight, no VRAM issues)
        llm = PosterPromptGenerator(
            base_url=self.config["llm_base_url"],
            api_key=self.config["llm_api_key"],
            model_name=self.config["llm_model_name"]
        )
        
        # 2.2 Generate Prompt
        # Note: phrases must be a list for the prompt generator
        phrases_list = [product_phrase]
        
        poster_prompt = llm.generate(
            slogan=selected_slogan, 
            phrases=phrases_list
        )
        
        if not poster_prompt:
            print("!! [Error] LLM failed to generate poster prompt. Aborting.")
            return

        print(f"   Visual Prompt: {poster_prompt[:100]}...") # Print preview


        # ---------------------------------------------------------
        # STEP 3: MS-Diffusion - Generate Image
        # ---------------------------------------------------------
        print("\n>> [Phase 3] Generating Poster Image...")
        
        # 3.1 Load Image Generation Models (Clean VRAM assumed)
        ms_generator = MSDiffusionGenerator(
            base_model_path=self.config["sd_base_path"],
            image_encoder_path=self.config["clip_path"],
            adapter_ckpt_path=self.config["ms_adapter_path"]
        )
        
        # 3.2 Execute Generation
        saved_paths = ms_generator.generate(
            image_path=image_path,
            prompt=poster_prompt,
            phrases=phrases_list, 
            save_dir=save_dir,
            save_name=product_phrase,
            num_samples=2,        # Generate 2 candidates
            steps=30
        )
        
        # 3.3 Unload Models (Optional cleanup)
        del ms_generator
        self._clean_memory()

        print(f"\n{'='*20} Pipeline Finished {'='*20}")
        print(f"Results saved in: {save_dir}/{product_phrase}")
        for p in saved_paths:
            print(f"- {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Auto Poster Pipeline')
    
    parser.add_argument('--vlm-model-path', type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help='Path to the VLM model')
    parser.add_argument('--llm-base-url', type=str, default="https://router.huggingface.co/v1",
                        help='Base URL for the LLM API')
    parser.add_argument('--llm-api-key', type=str, default=os.getenv("HF_API_KEY", "your_api_key_here"),
                        help='API key for the LLM')
    parser.add_argument('--llm-model-name', type=str, default="deepseek-ai/DeepSeek-V3:novita",
                        help='Name of the LLM model')
    parser.add_argument('--sd-base-path', type=str, default="tabilityai/stable-diffusion-xl-base-1.0",
                        help='Path to the SD base model')
    parser.add_argument('--clip-path', type=str, default="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                        help='Path to the CLIP model')
    parser.add_argument('--ms-adapter-path', type=str, default="doge1516/MS-Diffusion/ms_adapter.bin",
                        help='Path to the MS adapter checkpoint')
    
    parser.add_argument('--input-image-path', type=str, default="example/sneaker.jpeg",
                        help='Path to the input image')
    parser.add_argument('--output-directory', type=str, default="./final_posters",
                        help='Directory to save the output posters')
    
    parser.add_argument('--brand-tone', type=str, default="High-end, Energetic",
                        help='Brand tone for the slogan')
    parser.add_argument('--platform', type=str, default="Instagram Post",
                        help='Platform for the poster')
    parser.add_argument('--target-audience', type=str, default="Gen Z",
                        help='Target audience for the poster')
    parser.add_argument('--slogan-word-count', type=str, default="3-6 words",
                        help='Word count constraint for the slogan')
    
    args = parser.parse_args()

    CONFIG = {
        "vlm_model_path": args.vlm_model_path,
        "llm_base_url": args.llm_base_url,
        "llm_api_key": args.llm_api_key,
        "llm_model_name": args.llm_model_name,
        "sd_base_path": args.sd_base_path,
        "clip_path": args.clip_path,
        "ms_adapter_path": args.ms_adapter_path
    }

    user_requirements = {
        "brand_tone": args.brand_tone,
        "platform": args.platform,
        "target_audience": args.target_audience,
        "slogan_word_count": args.slogan_word_count
    }

    pipeline = AutoPosterPipeline(CONFIG)
    pipeline.run(
        image_path=args.input_image_path,
        user_requirements=user_requirements,
        save_dir=args.output_directory
    )
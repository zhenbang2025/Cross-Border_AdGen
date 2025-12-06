import os
import time
import torch
import gc
import argparse
from typing import Dict, Any
from PIL import Image
import trimesh

# --- Import 3D Generation Modules ---
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline


class Hunyuan3DPipeline:
    def __init__(self):
        """Initialize 3D generation pipeline with ALL default configurations"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = {
            "shapegen_model_path": "tencent/Hunyuan3D-2mv",
            "shapegen_subfolder": "hunyuan3d-dit-v2-mv-turbo",
            "shapegen_variant": "fp16",
            "texgen_model_path": "tencent/Hunyuan3D-2",
            "texgen_subfolder": "hunyuan3d-paint-v2-0-turbo",
            
            "remove_background": True,
            "inference_steps": 5,
            "octree_resolution": 380,
            "num_chunks": 20000,
            "random_seed": 12345
        }
        print(f">> Using device: {self.device}")
        print(">> All model/config parameters use default values")

    def _clean_memory(self):
        """Clean GPU memory to prevent OOM errors"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        print(">> [System] Memory cleaned")

    def _validate_image_path(self, path: str) -> bool:
        """Validate image path exists and is valid"""
        if not os.path.exists(path):
            print(f"!! Warning: Image file not found - {path}")
            return False
        try:
            Image.open(path).verify()
            return True
        except Exception as e:
            print(f"!! Error: Invalid image file - {path}, {str(e)}")
            return False

    def _load_and_preprocess_images(self, image_paths: Dict[str, str]) -> Dict[str, Image.Image]:
        """Load and preprocess images (background removal)"""
        processed_images = {}
        rembg = BackgroundRemover() if self.config["remove_background"] else None

        for view, path in image_paths.items():
            if not self._validate_image_path(path):
                raise FileNotFoundError(f"Cannot process {view} view: invalid image path")
            
            image = Image.open(path).convert("RGBA")
            if self.config["remove_background"] and image.mode == "RGB":
                image = rembg(image)
            
            processed_images[view] = image
            print(f">> Processed {view} view: {os.path.basename(path)}")
        
        return processed_images

    def run(self, input_views: Dict[str, str], output_dir: str = "./3d_output"):
        """
        Run full 3D generation pipeline with default settings
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f">> Output directory: {output_dir}")

        # Step 1: Preprocess images
        print("\n" + "="*40 + " Step 1: Preprocess Images " + "="*40)
        try:
            processed_images = self._load_and_preprocess_images(input_views)
        except Exception as e:
            print(f"!! Image preprocessing failed: {str(e)}")
            return

        # Step 2: Generate 3D mesh
        print("\n" + "="*40 + " Step 2: Generate 3D Mesh " + "="*40)
        start_time = time.time()
        try:
            shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self.config["shapegen_model_path"],
                subfolder=self.config["shapegen_subfolder"],
                variant=self.config["shapegen_variant"]
            ).to(self.device)
            shape_pipeline.enable_flashvdm()

            mesh = shape_pipeline(
                image=processed_images,
                num_inference_steps=self.config["inference_steps"],
                octree_resolution=self.config["octree_resolution"],
                num_chunks=self.config["num_chunks"],
                generator=torch.manual_seed(self.config["random_seed"]),
                output_type='trimesh'
            )[0]

            base_mesh_path = os.path.join(output_dir, "base_mesh.glb")
            mesh.export(base_mesh_path)
            print(f">> Base mesh saved to: {base_mesh_path}")
            print(f">> Mesh generation time: {time.time() - start_time:.2f}s")

        except Exception as e:
            print(f"!! Mesh generation failed: {str(e)}")
            return
        finally:
            del shape_pipeline
            self._clean_memory()

        # Step 3: Generate texture
        print("\n" + "="*40 + " Step 3: Generate Texture " + "="*40)
        start_time = time.time()
        try:
            tex_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                self.config["texgen_model_path"],
                subfolder=self.config["texgen_subfolder"]
            ).to(self.device)

            mesh = trimesh.load(base_mesh_path)
            textured_mesh = tex_pipeline(
                mesh, 
                image=list(processed_images.values())
            )

            textured_output_path = os.path.join(output_dir, "textured_mesh.glb")
            textured_mesh.export(textured_output_path)
            print(f">> Textured mesh saved to: {textured_output_path}")
            print(f">> Texture generation time: {time.time() - start_time:.2f}s")

        except Exception as e:
            print(f"!! Texture generation failed: {str(e)}")
            return
        finally:
            del tex_pipeline
            self._clean_memory()

        print("\n" + "="*40 + " Pipeline Completed " + "="*40)
        print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hunyuan 3D Generation Pipeline (All Default Configs)")

    parser.add_argument('--front-image', type=str, 
                        default='example/example_mv_images/1/front.png',
                        help='Path to front view image')
    parser.add_argument('--left-image', type=str, 
                        default='example/example_mv_images/1/left.png',
                        help='Path to left view image')
    parser.add_argument('--back-image', type=str, 
                        default='example/example_mv_images/1/back.png',
                        help='Path to back view image')

    args = parser.parse_args()

    input_views = {
        "front": args.front_image,
        "left": args.left_image,
        "back": args.back_image
    }

    # 
    print(">> Starting Hunyuan3D Pipeline")
    pipeline = Hunyuan3DPipeline()
    pipeline.run(input_views=input_views)
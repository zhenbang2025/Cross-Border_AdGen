import json
import re
import torch
from typing import Optional, Dict, Any, List
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers.image_utils import load_image

class MarketingContentGenerator:
    def __init__(self, model_path: str, device: str = None):
        """
        Initializes the model and processor.
        
        Args:
            model_path (str): Path to the pretrained model.
            device (str): Device to load the model on ('cuda', 'cpu', 'mps'). 
                          If None, automatically detects.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading model from {model_path} on {self.device}...")
        
        try:
            # Load the model with automatic precision mapping
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map=self.device
            )
            # Load the processor
            self.processor = AutoProcessor.from_pretrained(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Critical Error loading model: {e}")
            raise

    def _build_marketing_prompt(
        self,
        brand_tone: Optional[str] = None,
        platform: Optional[str] = None,
        target_audience: Optional[str] = None,
        task_goal: Optional[str] = None,
        slogan_word_count: Optional[str] = None,
        copy_word_count: Optional[str] = None,
        additional_info: Optional[str] = None,
    ) -> str:
        """
        Constructs the system prompt based on user constraints.
        """
        prompt_template = f"""
# ROLE
You are a specialized AI creative engine. Your only function is to analyze a product image and generate marketing copy based on a set of user-defined constraints.

# CRITICAL INSTRUCTIONS
1. Analyze the provided image and all user-defined constraints below.
2. If a constraint is marked as "Infer from image" or "Not specified", you must use your visual analysis of the image to make a smart deduction.
3. Your entire response MUST be a single, valid JSON object.
4. Do not include any text, explanations, or markdown fences like ```json before or after the JSON object. Your response must start with `{{` and end with `}}`.

# USER-DEFINED CONSTRAINTS
- **Brand Tone**: "{brand_tone or 'Infer from image'}"
- **Platform**: "{platform or 'General Social Media (Instagram/Facebook)'}"
- **Target Audience**: "{target_audience or 'Infer from image'}"
- **Task Goal**: "{task_goal or 'Drive product sales and increase engagement'}"
- **Slogan Word Count Range**: "{slogan_word_count or '3-10 words'}"
- **Marketing Copy Word Count Range**: "{copy_word_count or '25-60 words'}"
- **Additional Info**: "{additional_info or 'None'}"

# Output strictly in this JSON structure:
{{
  "slogans": [
    "Generate the first slogan here, adhering to the word count.",
    "Generate the second slogan here, offering a different angle.",
    "Generate the third slogan here, with another unique creative approach."
  ],
  "marketing_copies": [
    "Generate the first marketing copy here. It should be distinct, adhere to the word count, and align with the constraints.",
    "Generate the second marketing copy here, perhaps focusing more on an emotional benefit or a story.",
    "Generate the third marketing copy here, perhaps written as a direct call-to-action or for a specific ad format."
  ]
}}
"""
        return prompt_template.strip()

    def _extract_json_from_string(self, text: str) -> Optional[str]:
        """
        Robustly extracts the first valid JSON object string from a larger text.
        Handles cases where the model wraps output in markdown code blocks.
        """
        # Remove markdown code fences if present (e.g., ```json ... ```)
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```', '', text)

        # Regex to find content strictly between the outermost curly braces
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return None

    def generate(
        self,
        image_path: str,
        brand_tone: str = None,
        platform: str = None,
        target_audience: str = None,
        task_goal: str = None,
        slogan_word_count: str = None,
        copy_word_count: str = None,
        additional_info: str = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generates marketing slogans and copies based on an image and optional constraints.
        
        Returns:
            dict: Parsed JSON containing 'slogans' and 'marketing_copies', or None if failed.
        """
        
        # Verify image existence (load_image handles URLs and local paths, but explicit check is good)
        try:
            # We don't necessarily need to keep the PIL object, just ensure it loads
            _ = load_image(image_path)
        except Exception as e:
            print(f"Error: Could not load image at {image_path}. Details: {e}")
            return None

        # Build prompt
        prompt = self._build_marketing_prompt(
            brand_tone, platform, target_audience, task_goal,
            slogan_word_count, copy_word_count, additional_info
        )

        # Construct message payload for Qwen-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preprocessing
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision info (handles resizing, pixel values, etc.)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to the correct device
        inputs = inputs.to(self.device)

        # Inference
        # Increased token limit ensures complete JSON generation
        generated_ids = self.model.generate(**inputs, max_new_tokens=1000) 
        
        # Trim the input tokens from the output (standard Qwen-VL post-processing)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text_list = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        raw_response = output_text_list[0] if output_text_list else ""

        # print(f"\n--- Raw Model Response ---\n{raw_response}\n")

        # Parsing Logic
        json_string = self._extract_json_from_string(raw_response)
        
        if json_string:
            try:
                parsed_json = json.loads(json_string)
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"\n--- FAILED to Parse Extracted JSON ---")
                print(f"Error: {e}")
                print(f"Extracted String: {json_string}")
                return None
        else:
            print("\n--- No JSON object found in the model's response ---")
            print(f"Full Response: {raw_response}")
            return None

def print_results(result: Dict[str, Any], title: str):
    """Helper function to pretty-print results."""
    print(f"\n{'='*20} {title} {'='*20}")
    if result:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("Generation failed.")

if __name__ == '__main__':
    # Configuration
    MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct" 
    IMAGE_FILE = 'example/toy.jpeg'

    # Initialize Generator (Load model once)
    generator = MarketingContentGenerator(model_path=MODEL_PATH)

    # --- SCENARIO 1: Minimal User Input ---
    # print_results(
    #     generator.generate(
    #         image_path=IMAGE_FILE,
    #         slogan_word_count="8-15 words"
    #     ),
    #     "SCENARIO 1: Minimal Input"
    # )

    print("\n" * 2)

    # --- SCENARIO 2: Detailed User Input ---
    result_2 = generator.generate(
        image_path=IMAGE_FILE,
        brand_tone="fun",
        platform="Instagram Post",
        target_audience="Young people",
        task_goal="Promotion",
        slogan_word_count="6-10 words",
        copy_word_count="20-50 words"
    )
    
    print_results(result_2, "SCENARIO 2: Detailed Input")
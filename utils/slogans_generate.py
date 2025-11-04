import transformers
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def slogans_generate(image_path,prompt):
    # clip_model, preprocess = load("ViT-B/32", device=device)
    # image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    # image_features = clip_model.encode_image(image)

    # pipeline = transformers.pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")
    # prompt = f"基于商品特征：{image_features}，生成3条跨境电商宣传标语，突出卖点、简洁有力，适配英文平台"
    # slogans = pipeline(prompt, max_new_tokens=100)

    # Load images
    image = load_image("example/snaker.jpeg")
    # Initialize processor and model
    processor = AutoProcessor.from_pretrained("/home/rjiangas/models/SmolVLM-256M-Instruct")
    model = AutoModelForVision2Seq.from_pretrained(
        "/home/rjiangas/models/SmolVLM-256M-Instruct",
        torch_dtype=torch.bfloat16,
        # _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    ).to(DEVICE)


    # Create input messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        },
    ]

    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(DEVICE)
    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=1000)

    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print(generated_texts[0])
    import pdb;pdb.set_trace()
    return generated_texts[0]

if __name__ == '__main__':
    prompt = "Generate a poster description for the item in the image based on its content."
    slogans = slogans_generate('example/full-boots-2.png',prompt)
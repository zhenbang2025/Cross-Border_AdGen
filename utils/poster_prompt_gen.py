import os
from typing import Optional
from openai import OpenAI

class PosterPromptGenerator:
    def __init__(self, base_url: str, api_key: str, model_name: str):
        """
        Initializes the generator with API client configuration.

        Args:
            base_url (str): The base URL for the LLM API (e.g., Hugging Face, OpenAI).
            api_key (str): The API key for authentication.
            model_name (str): The specific model identifier to use.
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    def _get_system_prompt(self, slogan: str, phrase: str) -> str:
        """
        Constructs the structured system prompt required for the LLM.
        """
        system_prompt = f"""
Your Task: Based on slogan "{slogan}", you should intelligently analyzes its sentiment, style, and underlying meaning. It then matches or blends design elements from a diverse style library to generate a high-quality, professional text-to-image prompt for a typographic poster. You need to strictly follow the following rules:

# Product Phrase Binding
product phrase: {phrase}
The main product image used in Text-to-Image Model is represented by the phrase **{phrase}**.  
When generating the final prompt, you MUST explicitly include this exact word **{phrase}** to indicate
where the model should place the product in the scene.  
If there are additional creative assets, describe them generically (e.g. "light effect", "trail", "background")
but only the product phrase {phrase} corresponds to a provided image.

# Creative Style Library (for you to choose from)
1.  **Luminous Glow**: Neon contour lines, elegant and thin strokes, natural curves, transparent texture with inner glow effect. (Keywords: neon, glowing, outline, ethereal, luminous)
2.  **Industrial Grit**: Coarse metal texture, rusty and weathered details, embossed 3D structure, mechanical craftsmanship, rivet accents. (Keywords: industrial, metal, rust, grunge, mechanical, embossed)
3.  **Playful Doodle**: Casual and freehand lines, natural hand-drawn texture, cheerful and dynamic strokes, rounded edges, colorful gradients. (Keywords: doodle, hand-drawn, playful, kids, sketchy, colorful)
4.  **Sweet Pop**: Dreamy and girly aesthetic, soft and rounded letterforms, infused with bubblegum and candy elements, cute typography, star and heart decorations. (Keywords: cute, pop art, girly, candy, dreamy, pastel)
5.  **Anime Explosion**: Manga-style impact effect, high-tension extended lines, radiating and explosive strokes, creating strong visual impact. (Keywords: anime, manga, speed lines, action, dynamic, explosive)
6.  **Cyber Block**: High-contrast structure, geometric segmentation and reconstruction, neat and orderly arrangement, strong futuristic and tech feel. (Keywords: futuristic, tech, geometric, blocky, sci-fi)
7.  **Elegant Script**: Natural and flowing handwritten style, balanced and smooth lines, subtle ligatures, geometric beauty, clean start and end strokes. (Keywords: script, handwritten, elegant, calligraphy, signature)
8.  **Classic Ink Pen**: Graceful connected script, intricate thin lines, double-line layout, flowing and emotional typography. (Keywords: ink pen, classic, elegant, vintage, serif)
9.  **Mecha Sci-Fi**: Mechanical edges combined with streamlined design, neon accents, sharp corners, prominent metallic texture, integrated with circuit and chip patterns. (Keywords: mecha, sci-fi, cyberpunk, robotic, neon, metallic)
10. **Virtual Glitch**: Dark background, digitally deconstructed typography, glitch effect, sharp cutting lines, creating a futuristic tech visual. (Keywords: glitch, digital, deconstructed, vhs, cyberpunk, virtual)
11. **Retro Press**: Heavy and grainy font, old printing press effect, uneven ink distribution, slightly worn edges, strong nostalgic feel. (Keywords: retro, vintage, letterpress, grunge, old paper)
12. **Wild Calligraphy**: Energetic and unrestrained cursive style, "flying white" (feibai) brush technique, rhythmic variations, powerful and bold strokes. (Keywords: calligraphy, ink wash, sumi-e, expressive, bold brush)
13. **Gothic Elegance**: Modified Gothic style, elongated vertical proportions, sharp and pointed letterforms, rich decorative details, exuding mystery and solemnity. (Keywords: gothic, blackletter, medieval, ornate, mysterious)
14. **Kinetic Brush**: Brush strokes from thick to thin, tight and flowing structure, prominent "flying white" effect, high contrast emphasizing dynamic movement. (Keywords: kinetic, dynamic, brush stroke, action, motion)
15. **Deconstructed Bold**: Exaggerated and varied strokes, free-form ligatures, dislocated and deformed structure, showing a vibrant and rebellious visual effect. (Keywords: deconstructed, brutalist, bold, experimental)
16. **Minimalist Air**: Ultra-thin sans-serif design, ample white space, breathable letter spacing, modern Japanese design philosophy. (Keywords: minimalist, clean, thin, sans-serif, Japanese design)
17. **Icy Fracture**: Font edges with ice crackle effect, like frozen and shattered glass, sharp and cold strokes. (Keywords: ice, frozen, cracked, shattered, sharp, crystal)
18. **Licht Cut**: Typography composed of light-cutting planes, highlights shimmering with iridescent colors, blending mechanical feel with future tech. (Keywords: light effect, prismatic, iridescent, chrome, futuristic)
19. **Urban Graffiti**: Colorful graffiti art, thick and heavy outlines, layered and dynamic structure, a mix of handwritten and print styles. (Keywords: graffiti, street art, urban, spray paint, vibrant)
20. **Natural Woodcut**: Font edges with natural rough texture, strong wood carving feel, warm and rustic atmosphere. (Keywords: woodcut, rustic, natural, carved, organic)

# Intelligent Matching Rules
- **Short & Punchy Text (1-4 words)**: Match with high-impact styles like Industrial Grit, Anime Explosion, Mecha Sci-Fi, Icy Fracture.
- **Poetic & Graceful Text**: Match with artistic styles like Wild Calligraphy, Elegant Script, Minimalist Air, Classic Ink Pen.
- **Playful & Cute Text**: Match with lighthearted styles like Playful Doodle, Sweet Pop, Urban Graffiti.
- **Tech & Future-related Text**: Match with futuristic styles like Cyber Block, Mecha Sci-Fi, Virtual Glitch, Licht Cut.
- **Romantic & Emotional Text**: Match with styles like Luminous Glow, Classic Ink Pen, Elegant Script.
- **Retro & Nostalgic Text**: Match with styles like Retro Press, Natural Woodcut.

# Your Output Format(no more than 40 words; Follow the output format strictly, no extra content allowed)
`slogan="{slogan}", [Main Style Description], [Background Setting], [Typography & Composition], [Visual Effects & Details], [Color Palette], [Mood & Atmosphere], masterpiece, best quality, 8k.`

## Output Examples
### slogan: "Better is Temporary"; phrase: shoe
"Better is Temporary", Kinetic Brush style, minimalist dark background, a single shoe anchors the composition, dynamic typography with flying white effect, glowing energy trail, monochrome with electric blue accents, powerful and aspirational mood, masterpiece, best quality, 8k.
"""
        return system_prompt.strip()

    def generate(self, slogan: str, phrase: str) -> Optional[str]:
        """
        Takes a slogan and a product phrase, interacts with the LLM, 
        and returns a detailed image generation prompt.
        """
        system_message = self._get_system_prompt(slogan, phrase)
        
        # print(f"Debug: System message length: {len(system_message)} chars")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": system_message}
                ],
                temperature=0.7,  # Slight creativity boost
                max_tokens=2000
            )
            
            poster_prompt = response.choices[0].message.content
            
            print(f"\n{'='*20} Generated Poster Prompt {'='*20}")
            print(poster_prompt)
            print('='*64 + "\n")
            
            return poster_prompt
            
        except Exception as e:
            print(f"Error generating poster prompt: {e}")
            return None

# --- Main Execution Workflow ---
if __name__ == '__main__':
    
    # 1. Configuration
    API_KEY = os.getenv("HF_API_KEY", "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") 
    BASE_URL = "https://router.huggingface.co/v1"
    MODEL_NAME = "deepseek-ai/DeepSeek-V3:novita"

    # 2. Initialize Generator
    generator = PosterPromptGenerator(
        base_url=BASE_URL,
        api_key=API_KEY,
        model_name=MODEL_NAME
    )

    # 3. Define Input Data
    test_slogan = "Better is Temporary"
    product_phrase = "shoe"

    # 4. Generate the "Text Style" portion of the prompt
    text_style_prompt = generator.generate(test_slogan, product_phrase)

    if text_style_prompt:
        # 5. (Optional) Pipeline integration example
        # In a real scenario, `product_image_description` and `creative_element_prompt` 
        # would be generated by a VLM or logic defined elsewhere in your system.
        
        print(">> Next Step: Passing this prompt to Image Generation Model (e.g., MS-Diffusion)...")
        
        # Example pseudo-code for the next step:
        # generate_poster_with_ms_diffusion(
        #     product_image="path/to/image.png",
        #     poster_prompt=text_style_prompt,
        #     creative_element_prompt="glowing data streams flowing around the shoes"
        # )
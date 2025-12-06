import os
import io
import re
import glob
from typing import List, Optional, Dict
from enum import Enum

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
import yt_dlp
import whisper
import torch

# Load environment variables
load_dotenv()

app = FastAPI(title="EduComic Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client
try:
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Warning: Gemini Client failed to initialize. {e}")
    client = None

# --- CONSTANTS ---
UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "generated_comics"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class AgeGroup(str, Enum):
    TODDLER = "2-5"
    KID = "6-10"
    TEEN = "11+"

# --- HELPER FUNCTIONS (Program 1 Logic) ---

def combine_images_vertical(image_paths, output_path):
    """Combines images vertically (Standard Program 1 Logic)"""
    if not image_paths: return None
    try:
        images = [Image.open(path).convert("RGBA") for path in image_paths]
        widths, heights = zip(*(img.size for img in images))
        max_width = max(widths)
        total_height = sum(heights)
        
        combined_img = Image.new("RGBA", (max_width, total_height))
        y_offset = 0
        for img in images:
            combined_img.paste(img, (0, y_offset))
            y_offset += img.height
            
        combined_img.save(output_path, "PNG")
        return output_path
    except Exception as e:
        print(f"Error combining: {e}")
        return None

def split_pages(text: str) -> Dict[str, str]:
    """Splits text by [Page X] markers"""
    blocks = re.split(r"(?=\[Page\s+\d+\])", text)
    pages = {}
    for block in blocks:
        block = block.strip()
        if not block: continue
        m = re.match(r"\[Page\s+(\d+)\]", block)
        if m:
            key = f"page{m.group(1)}"
            pages[key] = block
    return pages

def get_educational_prompt(theme: str, content: str, num_pages: int, age_group: AgeGroup) -> str:
    """
    MODIFIED PROMPT:
    Uses the structure of Program 1 but injects Educational/Age constraints.
    """
    
    if age_group == AgeGroup.TODDLER:
        role = "You are an illustrator for nursery rhymes and toddler picture books."
        style = "Visual Style: Bright primary colors, flat vector art, thick outlines. Cute, rounded characters (animals or soft shapes). No scary elements."
        pacing = "Pacing: Very slow, 1-2 big panels per page."
        tone = "Tone: Joyous, musical, repetitive, very simple words."
    elif age_group == AgeGroup.KID:
        role = "You are a creator of popular Saturday Morning Cartoons."
        style = "Visual Style: Vibrant, energetic, dynamic poses, expressive faces. Relatable kid characters or superheroes."
        pacing = "Pacing: Dynamic, 3-4 panels per page."
        tone = "Tone: Fun, adventurous, exciting, jokes, fun facts."
    else: # TEEN
        role = "You are a professional Manga artist."
        style = "Visual Style: High-quality Manga/Anime style, detailed backgrounds, screen tones."
        pacing = "Pacing: Cinematic, 4-6 panels per page."
        tone = "Tone: Witty, cool, intellectual but accessible."

    output_format_lines = []
    for i in range(1, num_pages + 1):
        output_format_lines.append(f"[Page {i}]")
        if i == 1:
            output_format_lines.append("Title: (Catchy Title)")
        output_format_lines.append("→ Detailed description of the visual panels")
        output_format_lines.append("→ Dialogue/Captions")
        output_format_lines.append("")

    output_format = "\n".join(output_format_lines)

    return f"""
    {role}
    I am commissioning you to create a {num_pages}-page comic script to explain the concept: "{theme}".
    
    【Educational Context】
    {content[:4000]}

    【Strict Constraints for Age {age_group.value}】
    - {style}
    - {pacing}
    - {tone}
    - The explanation must be accurate but appropriate for the age.
    - Page 1 must introduce the topic clearly.
    - The final page must have a summary or "Lesson Learned".

    You must follow this exact format:
    【Output Format】
    {output_format}
    """

# --- VIDEO LOGIC (Program 2 Logic) ---

def download_audio(url: str, output_path: str):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3','preferredquality': '192'}],
        'outtmpl': output_path.replace('.mp3', ''),
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        base = output_path.replace('.mp3', '')
        possible_files = glob.glob(f"{base}*.mp3")
        return possible_files[0] if possible_files else None
    except Exception as e:
        print(f"DL Error: {e}")
        return None

def transcribe_audio_local(audio_path: str):
    # Use CPU to avoid the NumPy 2.0/MPS crash on Mac
    print("Loading Whisper (CPU mode)...")
    model = whisper.load_model('base', device="cpu") 
    result = model.transcribe(audio_path)
    return result['text']

# --- API ENDPOINTS ---

class ComicRequest(BaseModel):
    theme: str
    youtube_url: Optional[str] = None
    age_group: AgeGroup
    num_pages: int = 4

@app.post("/process-content")
async def process_content(request: ComicRequest):
    content_context = ""
    if request.youtube_url:
        print(f"Processing Video: {request.youtube_url}")
        file_id = re.sub(r'\W+', '', request.youtube_url[-11:])
        audio_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp3")
        
        if not os.path.exists(audio_path):
            downloaded_path = download_audio(request.youtube_url, audio_path)
            if not downloaded_path:
                raise HTTPException(status_code=400, detail="Failed to download audio")
            audio_path = downloaded_path
            
        try:
            transcript = transcribe_audio_local(audio_path)
            # Use Text Model for Summary
            summary_prompt = f"Extract the core educational concepts from this transcript suitable for a {request.age_group.value} year old:\n\n{transcript[:10000]}"
            summary_resp = client.models.generate_content(
                model="gemini-3-pro-preview", 
                contents=summary_prompt
            )
            content_context = summary_resp.text
        except Exception as e:
            print(f"Transcription error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        content_context = "General topic explanation."

    return {"context": content_context}

@app.post("/generate-comic")
async def generate_comic_endpoint(
    theme: str = Form(...),
    context: str = Form(...),
    age_group: AgeGroup = Form(...),
    num_pages: int = Form(...)
):
    try:
        # 1. Generate Plot (Text Model)
        print("Step 1: Generating Plot...")
        prompt = get_educational_prompt(theme, context, num_pages, age_group)
        
        plot_response = client.models.generate_content(
            model="gemini-3-pro-preview", 
            contents=prompt
        )
        plot_text = plot_response.text
        pages = split_pages(plot_text)
        
        # 2. Generate Images (Image Model - Program 1 Workflow)
        print("Step 2: Starting Image Generation Session...")
        
        # Create the Chat Session with IMAGE modalities (The Program 1 "Magic")
        chat = client.chats.create(
            model="gemini-3-pro-image-preview",
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )
        
        image_files = []
        generated_pages_pil = []
        
        for i in range(1, num_pages + 1):
            page_key = f"page{i}"
            if page_key not in pages: continue
            
            print(f"Generating Page {i}/{num_pages}...")
            page_prompt = pages[page_key]
            
            # Construct message: Prompt + (Previous Image if exists)
            message_parts = [page_prompt]
            if generated_pages_pil:
                # Add the previous image to keep consistency (Program 1 Logic)
                message_parts.append(generated_pages_pil[-1])
            
            try:
                response = chat.send_message(message_parts)
                
                # Check for valid image part
                if response.parts:
                    for part in response.parts:
                        if part.inline_data:
                            # Convert bytes to PIL Image
                            img = Image.open(io.BytesIO(part.inline_data.data))
                            
                            # Save locally
                            fname = f"comic_{theme[:5]}_{i}.png".replace(" ", "_")
                            fpath = os.path.join(OUTPUT_DIR, fname)
                            img.save(fpath)
                            
                            image_files.append(fpath)
                            generated_pages_pil.append(img)
                            break
            except Exception as e:
                print(f"Error on page {i}: {e}")
                # Try one retry or skip
                continue

        # 3. Combine Images
        print("Step 3: Combining...")
        if not image_files:
            raise HTTPException(status_code=500, detail="Failed to generate any images.")

        final_filename = f"final_{theme[:10]}_{age_group.value}.png".replace(" ", "_")
        final_path = os.path.join(OUTPUT_DIR, final_filename)
        combine_images_vertical(image_files, final_path)
        
        return FileResponse(final_path)

    except Exception as e:
        print(f"Critical Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
import os
import base64
import io
import json
import re
import requests
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Satellite Cloud Detection API")

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Azure OpenAI Config ---
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "GPT-4o-mini")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-07-01-preview")

# --- Setup LLM ---
try:
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_DEPLOYMENT_NAME,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_API_VERSION,
        model=AZURE_DEPLOYMENT_NAME,  # Explicitly set the model name
        temperature=0.1,  # Low temperature for consistent classification
        model_kwargs={
            "response_format": {"type": "json_object"}
        }
    )
    print("Azure OpenAI client initialized successfully")
    print(f"Using model: {AZURE_DEPLOYMENT_NAME}")
except Exception as e:
    print(f"Error initializing Azure OpenAI client: {e}")
    llm = None

# --- Output Schema ---
class CloudDetectionResponse(BaseModel):
    accuracy: float = Field(description="The confidence/accuracy of the result (0-100)")
    result: str = Field(description="The result of the classification: 'Clear' or 'Cloudy'")

# --- API Response Models ---
class PredictionResponse(BaseModel):
    prediction: str
    accuracy: float
    status: str
    message: str

def encode_image_to_base64(image_data: bytes) -> str:
    """Convert image bytes to base64 string"""
    return base64.b64encode(image_data).decode("utf-8")

def classify_image(image_base64: str) -> CloudDetectionResponse:
    """Classify satellite image using Azure OpenAI"""
    if not llm:
        raise HTTPException(status_code=500, detail="Azure OpenAI client not initialized")
    
    # --- Prompt Construction ---
    message = [
        {
            "role": "system",
            "content": """You are an expert satellite image analyst. Based on the satellite image provided, classify the scene as either: 
            'Clear' (no clouds or minimal cloud coverage) or 'Cloudy' (significant cloud coverage). 
            
            You must respond with a valid JSON object containing exactly these fields:
            {
                "result": "Clear" or "Cloudy",
                "accuracy": confidence_level_as_number_between_0_and_100
            }
            
            Analyze the image carefully and consider cloud density, coverage area, and visibility of ground features.
            Do not include any text outside the JSON object.""",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Classify this satellite image and return the result as JSON with 'result' and 'accuracy' fields.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ],
        },
    ]
    
    try:
        # Call the LLM with JSON mode
        response = llm.invoke(message)
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        print(f"Raw JSON response: {response_content}")
        
        # Parse the JSON response
        parsed_response = json.loads(response_content)
        
        result = CloudDetectionResponse(
            result=parsed_response.get('result', 'Unknown'),
            accuracy=float(parsed_response.get('accuracy', 0.0))
        )
        
        print(f"Prediction: {result.result}")
        print(f"Accuracy: {result.accuracy}%")
        return result
            
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response content: {response_content}")
        raise HTTPException(status_code=500, detail=f"Invalid JSON response: {str(e)}")
    except Exception as e:
        print(f"Error during classification: {e}")
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Satellite Cloud Detection API", "status": "running"}

@app.post("/classify-upload")
async def classify_uploaded_image(file: UploadFile = File(...)):
    """Classify an uploaded image file"""
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        
        # Validate image can be opened
        try:
            img = Image.open(io.BytesIO(image_data))
            img.verify()  # Verify it's a valid image
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Convert to base64
        image_base64 = encode_image_to_base64(image_data)
        
        # Classify image
        result = classify_image(image_base64)
        
        return PredictionResponse(
            prediction=result.result,
            accuracy=result.accuracy,
            status="success",
            message="Image classified successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing uploaded image: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/classify-url")
async def classify_image_from_url(image_url: str):
    """Classify an image from URL"""
    try:
        # Download image from URL
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Validate image
        try:
            img = Image.open(io.BytesIO(response.content))
            img.verify()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image URL or unsupported format")
        
        # Convert to base64
        image_base64 = encode_image_to_base64(response.content)
        
        # Classify image
        result = classify_image(image_base64)
        
        return PredictionResponse(
            prediction=result.result,
            accuracy=result.accuracy,
            status="success",
            message="Image classified successfully"
        )
        
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing URL image: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "azure_openai_configured": bool(AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY),
        "llm_initialized": llm is not None
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Satellite Cloud Detection API...")
    print(f"Azure OpenAI Endpoint: {AZURE_OPENAI_ENDPOINT}")
    print(f"Deployment: {AZURE_DEPLOYMENT_NAME}")
    uvicorn.run(app, host="0.0.0.0", port=8000)

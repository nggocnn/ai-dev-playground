# Satellite Cloud Detection Application

## Overview

This application provides AI-powered satellite image cloud detection using Azure OpenAI's GPT-4 Vision model. It consists of a Python FastAPI backend and a Next.js frontend, allowing users to upload satellite images and receive instant classification as either "Cloudy" or "Clear" with confidence scores.

## Features

- **Image Upload**: Upload satellite images directly from your device
- **URL Analysis**: Analyze images from web URLs
- **Mock Testing**: Test the system with predefined sample images
- **Real-time Results**: Instant cloud detection with confidence scores
- **Modern UI**: Clean, responsive web interface built with Next.js and Tailwind CSS
- **API-First**: RESTful API that can be integrated into other systems

## Architecture

```txt
satellite-cloud-detection/
├── backend/               # Python FastAPI backend
│   ├── main.py           # Main API application
│   ├── requirements.txt  # Python dependencies
│   └── .env             # Environment variables
├── frontend/             # Next.js frontend
│   ├── app/             # Next.js 13+ app directory
│   ├── package.json     # Node.js dependencies
│   └── *.config.js      # Configuration files
├── .env                 # Main environment file
└── README.md           # This file
```

## Prerequisites

- Python 3.8+
- Node.js 18+
- Azure OpenAI access with GPT-4 Vision deployment

## Setup Instructions

### 1. Environment Configuration

Create a `.env` file in the root directory with your Azure OpenAI credentials:

```env
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=your-gpt4-vision-deployment-name
AZURE_OPENAI_API_VERSION=2024-07-01-preview
```

### 2. Backend Setup (Python FastAPI)

1. **Navigate to backend directory:**

   ```bash
   cd backend
   ```

2. **Create virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Start the backend server:**

   ```bash
   python main.py
   ```

   The API will be available at `http://localhost:8000`

### 3. Frontend Setup (Next.js)

1. **Navigate to frontend directory:**

   ```bash
   cd frontend
   ```

2. **Install dependencies:**

   ```bash
   npm install
   ```

3. **Start the development server:**

   ```bash
   npm run dev
   ```

   The web application will be available at `http://localhost:3000`

## API Endpoints

### Health Check

- **GET** `/health` - Check API status and configuration

### Image Classification

- **POST** `/classify-upload` - Upload and classify an image file
- **POST** `/classify-url` - Classify an image from URL

### Testing

- **GET** `/test-mock-images` - Test with predefined sample images

## Usage Examples

### 1. Web Interface

1. Open `http://localhost:3000` in your browser
2. Upload an image file or enter an image URL
3. Click "Analyze" to get instant results
4. View prediction and confidence score

### 2. API Testing with cURL

**Upload file:**

```bash
curl -X POST "http://localhost:8000/classify-upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/satellite-image.jpg"
```

**Analyze URL:**

```bash
curl -X POST "http://localhost:8000/classify-url?image_url=https://example.com/satellite.jpg" \
  -H "accept: application/json"
```

**Test mock images:**

```bash
curl -X GET "http://localhost:8000/test-mock-images" \
  -H "accept: application/json"
```

### 3. Python Script Example

```python
import requests

# Test with URL
response = requests.post(
    "http://localhost:8000/classify-url",
    params={"image_url": "https://example.com/satellite.jpg"}
)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['accuracy']}%")
```

## Mock Images Testing

The application includes three predefined test images for quick demonstration:

1. **Cloudy sky** - Fluffy white clouds
2. **Clear sky** - Clear blue sky
3. **Partly cloudy** - Mixed conditions

Run the mock test via:

- Web interface: Click "Test with Mock Images"
- API: GET `/test-mock-images`

## Technical Implementation

### Backend Technologies

- **FastAPI**: Modern Python web framework for APIs
- **LangChain**: Integration with Azure OpenAI
- **Pydantic**: Data validation and serialization
- **PIL/Pillow**: Image processing
- **python-dotenv**: Environment variable management

### Frontend Technologies

- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **Heroicons**: Beautiful SVG icons
- **Axios**: HTTP client for API calls

### AI/ML Technologies

- **Azure OpenAI GPT-4 Vision**: Multimodal large language model
- **LangChain**: Structured output parsing
- **Base64 Encoding**: Image data transmission

## Prompt Engineering

The system uses carefully crafted prompts to ensure accurate classification:

```python
system_prompt = """You are an expert satellite image analyst. Based on the satellite image provided, classify the scene as either: 
'Clear' (no clouds or minimal cloud coverage) or 'Cloudy' (significant cloud coverage). 

Analyze the image carefully and provide:
1. A classification: either 'Clear' or 'Cloudy'
2. Your confidence level as a percentage (0-100)

Be precise and consider cloud density, coverage area, and visibility of ground features."""
```

## Error Handling

The application includes comprehensive error handling:

- **File validation**: Ensures uploaded files are valid images
- **URL validation**: Checks image URLs are accessible
- **API error handling**: Graceful handling of Azure OpenAI API errors
- **Frontend error states**: User-friendly error messages

## Performance Considerations

- **Image optimization**: Automatic base64 encoding for API transmission
- **Async processing**: Non-blocking API calls
- **Error boundaries**: Graceful degradation on failures
- **Loading states**: Clear feedback during processing

## LangChain Integration

The application leverages LangChain for:

1. **Azure OpenAI Integration**:

   ```python
   from langchain_openai import AzureChatOpenAI
   
   llm = AzureChatOpenAI(
       azure_endpoint=AZURE_OPENAI_ENDPOINT,
       azure_deployment=AZURE_DEPLOYMENT_NAME,
       api_key=AZURE_OPENAI_API_KEY,
       api_version=AZURE_API_VERSION,
   )
   ```

2. **Structured Output Parsing**:

   ```python
   from pydantic import BaseModel, Field
   
   class CloudDetectionResponse(BaseModel):
       accuracy: float = Field(description="Confidence (0-100)")
       result: str = Field(description="'Clear' or 'Cloudy'")
   
   llm_with_structured_output = llm.with_structured_output(CloudDetectionResponse)
   ```

3. **Multimodal Message Construction**:

   ```python
   message = [
       {"role": "system", "content": system_prompt},
       {
           "role": "user",
           "content": [
               {"type": "text", "text": user_prompt},
               {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
           ]
       }
   ]
   ```

## Troubleshooting

### Common Issues

1. **Azure OpenAI Authentication Error**
   - Verify API key and endpoint in `.env` file
   - Check deployment name matches your Azure resource

2. **Frontend Can't Connect to Backend**
   - Ensure backend is running on `localhost:8000`
   - Check CORS configuration allows `localhost:3000`

3. **Image Upload Fails**
   - Verify image file is valid format (JPG, PNG, GIF)
   - Check file size is reasonable (< 10MB)

4. **Dependencies Not Found**
   - Ensure virtual environment is activated (Python)
   - Run `npm install` for frontend dependencies

### Debug Mode

Enable debug logging by modifying the backend:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Knowledge and Experience Gained

### Technical Skills

1. **Multimodal AI Integration**: Working with vision-language models for image classification
2. **API Development**: Building robust FastAPI applications with proper error handling
3. **Frontend-Backend Integration**: Connecting React/Next.js with Python APIs
4. **LangChain Framework**: Leveraging structured outputs and multimodal capabilities

### AI/ML Concepts

1. **Prompt Engineering**: Crafting effective prompts for image classification tasks
2. **Confidence Scoring**: Understanding and presenting model confidence
3. **Base64 Encoding**: Efficient image data transmission over APIs
4. **Vision Model Applications**: Practical use of large vision-language models

### Development Best Practices

1. **Environment Configuration**: Secure API key management
2. **Error Handling**: Comprehensive error handling across the stack
3. **User Experience**: Building intuitive interfaces for AI applications
4. **API Design**: RESTful API patterns and documentation

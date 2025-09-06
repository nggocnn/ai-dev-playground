'use client'

import { useState, useRef } from 'react'
import axios from 'axios'
import { CloudIcon, SunIcon, PhotoIcon } from '@heroicons/react/24/outline'

interface PredictionResult {
  prediction: string
  accuracy: number
  status: string
  message: string
}

interface MockImageResult {
  image_id: number
  description: string
  url: string
  prediction: string
  accuracy: number
  status: string
  error?: string
}

const MOCK_IMAGES = [
  {
    url: "https://images.pexels.com/photos/53594/blue-clouds-day-fluffy-53594.jpeg",
    description: "Cloudy sky with fluffy white clouds"
  },
  {
    url: "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",
    description: "Clear blue sky"
  },
  {
    url: "https://images.pexels.com/photos/209831/pexels-photo-209831.jpeg",
    description: "Partly cloudy sky"
  }
]

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [imageUrl, setImageUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const API_BASE_URL = 'http://localhost:8000'

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      setResult(null)
      
      // Create preview URL
      const url = URL.createObjectURL(selectedFile)
      setPreviewUrl(url)
    }
  }

  const handleFileUpload = async () => {
    if (!file) return

    setLoading(true)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await axios.post(`${API_BASE_URL}/classify-upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setResult(response.data)
    } catch (error) {
      console.error('Error uploading file:', error)
      setResult({
        prediction: 'Error',
        accuracy: 0,
        status: 'error',
        message: 'Failed to classify image'
      })
    } finally {
      setLoading(false)
    }
  }

  const handleUrlSubmit = async () => {
    if (!imageUrl.trim()) return

    setLoading(true)
    setResult(null)
    setPreviewUrl(imageUrl)

    try {
      const response = await axios.post(`${API_BASE_URL}/classify-url`, null, {
        params: {
          image_url: imageUrl
        }
      })

      setResult(response.data)
    } catch (error) {
      console.error('Error classifying URL image:', error)
      setResult({
        prediction: 'Error',
        accuracy: 0,
        status: 'error',
        message: 'Failed to classify image from URL'
      })
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setFile(null)
    setImageUrl('')
    setResult(null)
    setPreviewUrl(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const getPredictionIcon = (prediction: string) => {
    if (prediction.toLowerCase().includes('cloudy')) {
      return <CloudIcon className="h-6 w-6 text-gray-600" />
    } else if (prediction.toLowerCase().includes('clear')) {
      return <SunIcon className="h-6 w-6 text-yellow-500" />
    }
    return null
  }

  const getPredictionColor = (prediction: string) => {
    if (prediction.toLowerCase().includes('cloudy')) {
      return 'text-gray-600 bg-gray-100'
    } else if (prediction.toLowerCase().includes('clear')) {
      return 'text-yellow-700 bg-yellow-100'
    }
    return 'text-red-600 bg-red-100'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-800 mb-4">
              🛰️ Satellite Cloud Detection
            </h1>
            <p className="text-lg text-gray-600">
              AI-powered cloud detection using Azure OpenAI Vision
            </p>
          </div>

          {/* Main Content */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
            <div className="grid md:grid-cols-2 gap-8">
              {/* Input Section */}
              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-gray-800 mb-4">
                  Upload Image
                </h2>

                {/* File Upload */}
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6">
                  <div className="text-center">
                    <PhotoIcon className="mx-auto h-12 w-12 text-gray-400" />
                    <div className="mt-4">
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*"
                        onChange={handleFileChange}
                        className="hidden"
                        id="file-upload"
                      />
                      <label
                        htmlFor="file-upload"
                        className="cursor-pointer bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors"
                      >
                        Choose Image File
                      </label>
                      <p className="mt-2 text-sm text-gray-500">
                        PNG, JPG, GIF up to 10MB
                      </p>
                    </div>
                  </div>
                  {file && (
                    <div className="mt-4 text-center">
                      <p className="text-sm text-gray-600">Selected: {file.name}</p>
                      <button
                        onClick={handleFileUpload}
                        disabled={loading}
                        className="mt-2 bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg transition-colors disabled:opacity-50"
                      >
                        {loading ? 'Analyzing...' : 'Analyze Image'}
                      </button>
                    </div>
                  )}
                </div>

                {/* URL Input */}
                <div className="space-y-3">
                  <h3 className="text-lg font-medium text-gray-700">Or use Image URL</h3>
                  <div className="flex space-x-2">
                    <input
                      type="url"
                      value={imageUrl}
                      onChange={(e) => setImageUrl(e.target.value)}
                      placeholder="https://example.com/satellite-image.jpg"
                      className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <button
                      onClick={handleUrlSubmit}
                      disabled={loading || !imageUrl.trim()}
                      className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors disabled:opacity-50"
                    >
                      {loading ? 'Analyzing...' : 'Analyze'}
                    </button>
                  </div>
                </div>

                {/* Reset Button */}
                <button
                  onClick={resetForm}
                  className="w-full bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  Reset
                </button>
              </div>

              {/* Preview and Results Section */}
              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-gray-800 mb-4">
                  Results
                </h2>

                {/* Image Preview */}
                {previewUrl && (
                  <div className="border rounded-lg overflow-hidden">
                    <img
                      src={previewUrl}
                      alt="Preview"
                      className="w-full h-64 object-cover"
                    />
                  </div>
                )}

                {/* Loading Spinner */}
                {loading && (
                  <div className="flex items-center justify-center py-8">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
                    <span className="ml-3 text-lg text-gray-600">Analyzing image...</span>
                  </div>
                )}

                {/* Single Result */}
                {result && !loading && (
                  <div className="bg-gray-50 rounded-lg p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-medium text-gray-800">Prediction Result</h3>
                      {getPredictionIcon(result.prediction)}
                    </div>
                    <div className={`inline-block px-4 py-2 rounded-full text-sm font-medium ${getPredictionColor(result.prediction)}`}>
                      {result.prediction}
                    </div>
                    <p className="mt-2 text-gray-600">
                      Confidence: <span className="font-semibold">{result.accuracy.toFixed(1)}%</span>
                    </p>
                    {result.status === 'error' && (
                      <p className="mt-2 text-red-600 text-sm">{result.message}</p>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Mock Results removed - endpoint no longer available */}
        </div>
      </div>
    </div>
  )
}

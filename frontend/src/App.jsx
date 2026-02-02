import React, { useState } from 'react';
import axios from 'axios';
import Header from './components/Header';
import Footer from './components/Footer';
import UploadForm from './components/UploadForm';
import ResultDisplay from './components/ResultDisplay';
import './index.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null); // Clear previous results
    setError(null);
  };

  const handlePredict = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setResult(response.data);
      } else {
        setError('Prediction failed. Please try again.');
      }
    } catch (err) {
      console.error(err);
      setError('An error occurred while connecting to the server.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="font-roboto min-h-screen flex flex-col items-center bg-gray-50">
      <Header />

      <main
        className="w-[90%] max-w-[1000px] mt-[120px] mb-[60px] p-5 rounded-[15px] shadow-[0_4px_15px_rgb(30,64,208)] overflow-auto"
        style={{
          background: 'linear-gradient(rgba(255, 255, 255, 0.6), rgba(129, 32, 32, 0.4))',
          height: 'calc(100vh - 180px)'
        }}
      >
        <div className="flex flex-col gap-5">
          <UploadForm
            onFileSelect={handleFileSelect}
            previewUrl={previewUrl}
            onPredict={handlePredict}
            isLoading={isLoading}
          />

          {error && <div className="text-red-600 font-bold text-center">{error}</div>}

          {result && (
            <div className="w-full p-5 box-border rounded-10 shadow-[0_0_10px_rgb(97,91,91)] overflow-auto bg-[#526878] animate-[slideInRight_0.5s_ease-out]">
              <ResultDisplay
                prediction={result.prediction}
                cause={result.cause}
                pesticide={result.pesticide}
              />
            </div>
          )}
        </div>
      </main>

      <Footer />
    </div>
  );
}

export default App;

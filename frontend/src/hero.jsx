import React, { useState } from "react";
import myImage from "./diagram.png"; // Import the image file

function Hero({ onNavigateToSlides }) {
  const [file, setFile] = useState(null);
  const [messageVisible, setMessageVisible] = useState(false);
  const [localTrainingVisible, setLocalTrainingVisible] = useState(false);
  const [allDoneVisible, setAllDoneVisible] = useState(false);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = (event) => {
    event.preventDefault();

    if (!file) {
      alert("Please select a file to upload.");
      return;
    }

    setMessageVisible(true);

    setTimeout(() => {
      setMessageVisible(false);
      setLocalTrainingVisible(true);
    }, 2000);

    setTimeout(() => {
      setLocalTrainingVisible(false);
      setAllDoneVisible(true);
    }, 4000);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 bg-gradient-to-r from-blue-50 via-blue-100 to-blue-200 dark:from-gray-900 dark:to-gray-800 transition-colors duration-500">
      <div className="text-center">
        <h1 className="text-5xl font-extrabold text-gray-900 dark:text-gray-100 mb-4">
          MYCELIUM
        </h1>
        <h2 className="text-3xl font-semibold text-gray-800 dark:text-gray-200 mb-6">
          ML Decentralized & Slide Platform
        </h2>
        <img
          src={myImage}
          alt="Mycelium Architecture"
          className="w-64 h-auto object-cover rounded-lg shadow-md mb-8"
        />
        
        {/* Platform Options */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8 max-w-2xl">
          {/* ML Training Section */}
          <div className="bg-white/80 dark:bg-gray-800/80 p-6 rounded-lg shadow-lg">
            <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
              ML Training
            </h3>
            <p className="text-lg text-gray-700 dark:text-gray-300 mb-4">
              Upload your CSV file to get started with decentralized model training.
            </p>
            <form onSubmit={handleSubmit} className="flex flex-col items-center">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="mb-4 p-2 border border-gray-300 rounded-lg dark:border-gray-700"
              />
              <button
                type="submit"
                className="bg-blue-600 text-white py-2 px-6 rounded-lg shadow-md hover:bg-blue-700 transition-colors duration-300"
              >
                Start ML Training
              </button>
            </form>
          </div>

          {/* Slide Editor Section */}
          <div className="bg-white/80 dark:bg-gray-800/80 p-6 rounded-lg shadow-lg">
            <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
              Slide Editor
            </h3>
            <p className="text-lg text-gray-700 dark:text-gray-300 mb-4">
              Create presentations with our LLM-integrated slide editor platform.
            </p>
            <button
              onClick={onNavigateToSlides}
              className="bg-green-600 text-white py-2 px-6 rounded-lg shadow-md hover:bg-green-700 transition-colors duration-300"
            >
              Open Slide Editor
            </button>
          </div>
        </div>

        <div className="mt-8">
          {messageVisible && (
            <h2 className="text-2xl font-medium text-blue-600 dark:text-blue-400 animate-fadeIn">
              Decentralized model training happening now...
            </h2>
          )}
          {localTrainingVisible && (
            <h2 className="text-2xl font-medium text-blue-600 dark:text-blue-400 animate-fadeIn">
              Local training on decentralized nodes...
            </h2>
          )}
          {allDoneVisible && (
            <h2 className="text-2xl font-medium text-green-600 dark:text-green-400 animate-fadeIn">
              Global model updated!
            </h2>
          )}
        </div>
      </div>
    </div>
  );
}

export default Hero;

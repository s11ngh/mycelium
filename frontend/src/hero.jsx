import React, { useState } from "react";
import myImage from "./diagram.png"; // Import the image file

function Hero() {
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
          ML Decentralized
        </h2>
        <img
          src={myImage}
          alt="Mycelium Architecture"
          className="w-64 h-auto object-cover rounded-lg shadow-md mb-8"
        />
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-8">
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
            Upload
          </button>
        </form>
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

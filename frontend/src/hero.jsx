import React, { useState } from "react";

function Hero() {
  const [file, setFile] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = (event) => {
    event.preventDefault();

    if (!file) {
      alert("Please select a file to upload.");
      return;
    }

    // Implement your file upload logic here

    alert(`File selected: ${file.name}`);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 bg-gray-100 dark:bg-gray-800">
      <h1 className="text-4xl font-bold text-gray-900 dark:text-gray-100 mb-4">
        Decentralizing AI
      </h1>
      <p className="text-lg text-gray-700 dark:text-gray-300 mb-8">
        Upload your CSV file to get started
      </p>
      <form onSubmit={handleSubmit} className="flex flex-col items-center">
        <input
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          className="mb-4"
        />
        <button
          type="submit"
          className="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600"
        >
          Upload
        </button>
      </form>
    </div>
  );
}

export default Hero;

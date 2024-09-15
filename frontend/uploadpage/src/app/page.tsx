"use client"; // Ensure this page is a Client Component

import { useState } from "react";
import Image from "next/image";

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    setSelectedFile(file);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        alert("File uploaded successfully");
        setSelectedFile(null); // Clear file input
      } else {
        alert("File upload failed");
      }
    } catch (error) {
      console.error("An error occurred:", error);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-8 bg-background text-foreground">
      <main className="w-full max-w-lg p-6 bg-white rounded-lg shadow-lg dark:bg-gray-800">
        <h1 className="text-2xl font-bold mb-4 text-center">Upload CSV File</h1>
        <p className="mb-6 text-center text-gray-600 dark:text-gray-300">
          Choose a CSV file to upload. The file will be processed and saved on
          the server.
        </p>
        <input
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          className="mb-4 w-full px-3 py-2 border border-gray-300 rounded-md dark:border-gray-600"
        />
        <button
          onClick={handleUpload}
          className="w-full py-2 px-4 bg-blue-500 text-white rounded-md hover:bg-blue-600 dark:bg-blue-700 dark:hover:bg-blue-600"
        >
          Upload
        </button>
      </main>
      <footer className="mt-8 text-center text-gray-600 dark:text-gray-300">
        <p>© {new Date().getFullYear()} Your Company</p>
      </footer>
    </div>
  );
}

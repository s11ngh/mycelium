import React, { useState } from "react";
import "./App.css";
import Hero from "./hero";
import SlideEditor from "./components/SlideEditor";

function App() {
  const [currentView, setCurrentView] = useState('home'); // 'home' or 'slides'

  const navigateToSlides = () => {
    setCurrentView('slides');
  };

  const navigateToHome = () => {
    setCurrentView('home');
  };

  return (
    <div className="App">
      {currentView === 'home' && (
        <Hero onNavigateToSlides={navigateToSlides} />
      )}
      {currentView === 'slides' && (
        <div>
          <div className="bg-white border-b border-gray-300 p-2">
            <button
              onClick={navigateToHome}
              className="bg-gray-600 text-white px-3 py-1 rounded hover:bg-gray-700 transition-colors"
            >
              ← Back to Home
            </button>
          </div>
          <SlideEditor />
        </div>
      )}
    </div>
  );
}

export default App;

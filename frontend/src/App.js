import React, { useState } from "react";
import "./App.css";
import Upload from "./components/upload";
import LoginScreen from "./components/login";

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  return (
    <div className="app-container">
      {/* Background animated elements for dynamic design */}
      <div className="bg-shape shape1"></div>
      <div className="bg-shape shape2"></div>
      <div className="bg-shape shape3"></div>

      {/* Nature Vibe Background Elements */}
      <div className="nature-overlay">
        {[...Array(10)].map((_, i) => (
          <div key={`leaf-${i}`} className="leaf"></div>
        ))}
        {[...Array(10)].map((_, i) => (
          <div key={`firefly-${i}`} className="firefly"></div>
        ))}
      </div>

      <div className="content-wrapper">
        <header className="app-header">
          <h1 className="main-title">
            <span className="icon">🍎</span> Apple Leaf Disease Detector
          </h1>
          <p className="subtitle">Instant crop health analysis using advanced AI models.</p>
        </header>

        {isLoggedIn ? (
          <Upload />
        ) : (
          <LoginScreen onLogin={() => setIsLoggedIn(true)} />
        )}
      </div>
    </div>
  );
}

export default App;
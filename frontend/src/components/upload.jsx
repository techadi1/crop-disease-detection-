import React, { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';
import "./upload.css";

export default function UploadScreen() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  // API State
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const reportRef = useRef(null);

  const handleImage = (file) => {
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) handleImage(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleImage(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const analyzeLeaf = async () => {
    if (!image) return;
    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Failed connecting to analysis server");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "An error occurred during analysis");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const downloadPDF = async () => {
    const element = reportRef.current;
    const canvas = await html2canvas(element, { scale: 2, useCORS: true, backgroundColor: "#0f172a" });
    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF('p', 'mm', 'a4');

    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

    pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
    pdf.save("apple-leaf-report.pdf");
  };

  // Modern vibrant colors for the chart bars
  const chartColors = ['#f87171', '#fb923c', '#fbbf24', '#34d399'];

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9, y: 30 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.6, type: "spring", bounce: 0.4 }}
      className="glass-card upload-container-wide"
    >
      <div className="glass-card-inner">
        <h2 className="card-title">Analysis Center</h2>

        <div className="upload-section">
          <motion.label
            className={`upload-dropzone ${isDragging ? 'dragging' : ''} ${preview ? 'has-image' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            whileHover={!preview ? { scale: 1.02 } : {}}
            whileTap={!preview ? { scale: 0.98 } : {}}
          >
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden-input"
            />

            <AnimatePresence mode="wait">
              {!preview ? (
                <motion.div
                  key="upload-prompt"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  className="upload-prompt"
                >
                  <div className="upload-icon-wrapper">
                    <svg className="upload-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
                    </svg>
                  </div>
                  <p className="prompt-text">Drag & Drop Image</p>
                  <p className="prompt-subtext">or click to browse local files</p>
                </motion.div>
              ) : (
                <motion.div
                  key="preview"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="preview-container"
                >
                  <img src={preview} alt="Leaf preview" className="preview-image" />
                  <div className="preview-overlay">
                    <span>Change Image</span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.label>
        </div>

        {/* Results Area */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="error-message"
              style={{ color: "#ef4444", marginBottom: "15px", textAlign: "center" }}
            >
              ⚠️ {error}
            </motion.div>
          )}

          {result && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="dashboard-wrapper"
              style={{ width: "100%", marginBottom: "20px" }}
            >
              {/* Dashboard content that will be downloaded via PDF */}
              <div ref={reportRef} className="report-canvas" style={{ padding: "20px", background: "rgba(15, 23, 42, 0.7)", borderRadius: "16px", border: "1px solid rgba(255,255,255,0.1)" }}>

                <div style={{ textAlign: "center", marginBottom: "20px" }}>
                  <h3 style={{ color: "#a7f3d0", fontSize: "1.2rem", marginBottom: "8px" }}>Diagnosis Complete</h3>
                  <p style={{ color: "#fff", fontSize: "1.6rem", fontWeight: "bold", wordBreak: "break-word" }}>
                    {result.prediction.replace(/___/g, " - ").replace(/_/g, " ")}
                  </p>
                  <p style={{ color: "#6ee7b7", fontSize: "1rem", marginTop: "5px" }}>
                    Peak Confidence: <span style={{ fontWeight: 'bold' }}>{result.confidence?.toFixed(2)}%</span>
                  </p>
                </div>

                {/* Chart UI */}
                <div style={{ width: '100%', height: 200, marginTop: "10px" }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={result.probabilities} margin={{ top: 5, right: 20, left: -20, bottom: 5 }}>
                      <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 10 }} interval={0} />
                      <YAxis tick={{ fill: '#94a3b8' }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#fff' }}
                        itemStyle={{ color: '#34d399' }}
                        cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                      />
                      <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                        {result.probabilities.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={chartColors[index % chartColors.length]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Download Button */}
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={downloadPDF}
                className="download-pdf-btn"
                style={{
                  width: "100%",
                  padding: "12px",
                  marginTop: "15px",
                  borderRadius: "12px",
                  border: "1px solid rgba(59, 130, 246, 0.5)",
                  background: "rgba(59, 130, 246, 0.1)",
                  color: "#60a5fa",
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                  gap: "8px",
                  cursor: "pointer",
                  fontWeight: "600",
                  transition: "all 0.3s"
                }}
              >
                <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Download Graph PDF Dashboard
              </motion.button>
            </motion.div>
          )}
        </AnimatePresence>

        <motion.button
          onClick={analyzeLeaf}
          whileHover={preview && !isAnalyzing ? { scale: 1.05, boxShadow: "0 10px 25px -5px rgba(16, 185, 129, 0.5)" } : {}}
          whileTap={preview && !isAnalyzing ? { scale: 0.95 } : {}}
          className="analyze-button"
          disabled={!preview || isAnalyzing}
        >
          {isAnalyzing ? "Analyzing AI Model..." : preview ? "Analyze Leaf Health" : "Waiting for Image..."}
          {!isAnalyzing && (
            <svg className="arrow-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          )}
        </motion.button>

      </div>
    </motion.div>
  );
}
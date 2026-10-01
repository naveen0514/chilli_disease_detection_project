import React, {useState} from "react";
import {createRoot} from "react-dom/client";
import "./style.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const choose = (e) => {
    const f = e.target.files[0];
    setFile(f);
    setResult(null);
    setError("");
    if (f) setPreview(URL.createObjectURL(f));
  };

  const predict = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);

    const form = new FormData();
    form.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: form
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Prediction failed");
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="card">
        <h1>🌶️ Chilli Disease Detection</h1>
        <p>Upload a chilli leaf image and let the trained deep-learning model classify it.</p>

        <input type="file" accept="image/*" onChange={choose}/>

        {preview && <img className="preview" src={preview} />}

        <button disabled={!file || loading} onClick={predict}>
          {loading ? "Predicting..." : "Detect Disease"}
        </button>

        {error && <div className="error">{error}</div>}

        {result && (
          <div className="result">
            <h2>{result.disease}</h2>
            <p>Confidence: <b>{result.confidence}%</b></p>
          </div>
        )}
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);

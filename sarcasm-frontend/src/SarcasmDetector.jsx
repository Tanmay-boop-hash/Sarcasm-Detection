// SarcasmDetector.jsx
import { useState } from "react";

const API_URL = "https://Tanmay-24-sarcasm-detector-api.hf.space/predict";

export default function SarcasmDetector() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const examples = [
    "local man discovers sarcasm, immediately regrets everything",
    "study finds humans still bad at parking",
    "scientists confirm water is, in fact, wet",
  ];

  const handleSubmit = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error("Request failed");
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError("Couldn't reach the model. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 480, margin: "40px auto", fontFamily: "sans-serif", fontWeight: "bolder" }}>
      <h2>Sarcasm Detector</h2>
      <p style={{ fontSize: 13, color: "#8d8989" }}>
        Trained on news headlines — works best on headline-style text.
      </p>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type a headline-style sentence..."
        rows={3}
        style={{ width: "100%", padding: 10, fontSize: 14 }}
      />

      <button
        onClick={handleSubmit}
        disabled={loading}
        style={{ marginTop: 10, padding: "8px 16px", cursor: "pointer" }}
      >
        {loading ? "Checking..." : "Detect"}
      </button>

      <div style={{ marginTop: 10 }}>
        {examples.map((ex) => (
          <button
            key={ex}
            onClick={() => setText(ex)}
            style={{
              marginRight: 6,
              marginBottom: 6,
              fontSize: 12,
              padding: "4px 8px",
              cursor: "pointer",
            }}
          >
            {ex}
          </button>
        ))}
      </div>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {result && (
        <div style={{ marginTop: 20, padding: 16, border: "1px solid #ddd", borderRadius: 8 }}>
          <strong>{result.is_sarcastic ? "Sarcastic 🙃" : "Not sarcastic"}</strong>
          <p>Confidence: {(result.confidence * 100).toFixed(1)}%</p>
          <div style={{ fontSize: 13, color: "#8c8989" }}>
            <div>Not sarcastic: {(result.probabilities.not_sarcastic * 100).toFixed(1)}%</div>
            <div>Sarcastic: {(result.probabilities.sarcastic * 100).toFixed(1)}%</div>
          </div>
        </div>
      )}
    </div>
  );
}
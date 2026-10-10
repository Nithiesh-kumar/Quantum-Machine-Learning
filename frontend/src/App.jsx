import React, { useEffect, useState } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import SamplePresets, { PATIENT_PRESETS } from './components/SamplePresets';
import Wizard from './components/Wizard';
import ResultScreen from './components/ResultScreen';
import ModelInsights from './components/ModelInsights';
import { AlertCircle, RefreshCw, ShieldAlert } from 'lucide-react';

const DEFAULT_PATIENT = {
  age: 58.0,
  sex: 1,
  cp: 4,
  trestbps: 140.0,
  chol: 240.0,
  fbs: 0,
  restecg: 1,
  thalach: 145.0,
  exang: 1,
  oldpeak: 1.8,
  slope: 2,
  ca: 1.0,
  thal: 7.0,
};

export default function App() {
  const [activeTab, setActiveTab] = useState('assessment');
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('cardioquantum_theme') || 'light';
  });

  const [patientData, setPatientData] = useState(() => {
    try {
      const saved = localStorage.getItem('cardioquantum_patient');
      return saved ? JSON.parse(saved) : DEFAULT_PATIENT;
    } catch {
      return DEFAULT_PATIENT;
    }
  });

  const [currentStep, setCurrentStep] = useState(1);
  const [activePreset, setActivePreset] = useState(null);
  const [lastSaved, setLastSaved] = useState(null);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiOnline, setApiOnline] = useState(true);
  const [apiError, setApiError] = useState(null);

  // Apply theme to document
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('cardioquantum_theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  };

  // Check API health on mount
  const checkHealth = async () => {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        setApiOnline(true);
        setApiError(null);
      } else {
        setApiOnline(false);
      }
    } catch {
      setApiOnline(false);
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  // Autosave patient data
  const handleFieldChange = (field, value) => {
    setPatientData((prev) => {
      const updated = { ...prev, [field]: value };
      try {
        localStorage.setItem('cardioquantum_patient', JSON.stringify(updated));
        const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        setLastSaved(timeStr);
      } catch {
        // LocalStorage access handling
      }
      return updated;
    });
  };

  // Preset Selection
  const handleSelectPreset = (preset) => {
    setPatientData(preset.data);
    setActivePreset(preset.id);
    localStorage.setItem('cardioquantum_patient', JSON.stringify(preset.data));
    setLastSaved(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
  };

  // Handle assessment submission
  const handleSubmitAssessment = async ({ model_name = 'production', compare = true }) => {
    setLoading(true);
    setApiError(null);

    const payload = {
      ...patientData,
      model_name,
      compare,
    };

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || errJson.message || `API server returned status ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
      setApiOnline(true);

      // Smooth scroll to results
      setTimeout(() => {
        const resultsEl = document.getElementById('results-section');
        if (resultsEl) {
          resultsEl.scrollIntoView({ behavior: 'smooth' });
        }
      }, 100);
    } catch (err) {
      setApiError(err.message);
      setApiOnline(false);
    } finally {
      setLoading(false);
    }
  };

  const handleStartClick = () => {
    setActiveTab('assessment');
    const formEl = document.getElementById('wizard-form');
    if (formEl) {
      formEl.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleReset = () => {
    setPatientData(DEFAULT_PATIENT);
    setCurrentStep(1);
    setResult(null);
    setActivePreset(null);
    localStorage.removeItem('cardioquantum_patient');
  };

  return (
    <div className="app-container">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        theme={theme}
        toggleTheme={toggleTheme}
        apiOnline={apiOnline}
      />

      {/* API-Down Fallback Notification Banner */}
      {!apiOnline && (
        <div className="error-banner" role="alert">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <AlertCircle size={20} />
            <div>
              <strong>FastAPI Service Disconnected:</strong> Cannot reach backend server at{' '}
              <code>http://127.0.0.1:8000</code>. Please check that Uvicorn is active.
            </div>
          </div>
          <button
            type="button"
            className="btn-secondary"
            onClick={checkHealth}
            style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem', border: 'none' }}
          >
            <RefreshCw size={14} />
            <span>Retry Connection</span>
          </button>
        </div>
      )}

      {/* Main Tab Content */}
      {activeTab === 'assessment' ? (
        <main>
          <Hero onStartClick={handleStartClick} />

          <SamplePresets
            activePreset={activePreset}
            onSelectPreset={handleSelectPreset}
            lastSaved={lastSaved}
          />

          <Wizard
            patientData={patientData}
            onChange={handleFieldChange}
            onReset={handleReset}
            onSubmit={handleSubmitAssessment}
            loading={loading}
            currentStep={currentStep}
            setCurrentStep={setCurrentStep}
          />

          {/* Results Screen */}
          {result && (
            <ResultScreen
              result={result}
              patientData={patientData}
              onEdit={() => {
                const el = document.getElementById('wizard-form');
                if (el) el.scrollIntoView({ behavior: 'smooth' });
              }}
              onReset={handleReset}
            />
          )}
        </main>
      ) : (
        <main>
          <div style={{ textAlign: 'center', padding: '1.5rem 0 2rem' }}>
            <h1 style={{ fontSize: '2.25rem', marginBottom: '0.5rem' }}>Model Architecture &amp; Empirical Benchmarks</h1>
            <p style={{ color: 'var(--text-secondary)', maxWidth: '650px', margin: '0 auto' }}>
              Detailed metrics across 5-seed cross-validation, quantum circuit topologies, and ROC curve analyses.
            </p>
          </div>
          <ModelInsights />
        </main>
      )}

      {/* Footer */}
      <footer style={{ marginTop: '4rem', paddingTop: '2rem', borderTop: '1px solid var(--border-subtle)', textAlign: 'center', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
        <p>
          <strong>CardioQuantum</strong> • Educational &amp; Research Prototype • Built with PennyLane, Qiskit, PyTorch, Scikit-Learn, and FastAPI.
        </p>
      </footer>
    </div>
  );
}

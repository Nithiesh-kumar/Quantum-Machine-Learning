import React, { useState } from 'react';
import {
  Download,
  RotateCcw,
  Sparkles,
  ShieldAlert,
  ShieldCheck,
  Cpu,
  Layers,
  ChevronDown,
  ChevronUp,
  FileText,
  Printer,
} from 'lucide-react';

export default function ResultScreen({
  result,
  patientData,
  onEdit,
  onReset,
}) {
  const [showFullComparison, setShowFullComparison] = useState(false);

  if (!result) return null;

  const prob = result.risk_probability;
  const percentage = Math.round(prob * 100);
  const isHigh = prob > 0.65;
  const isModerate = prob >= 0.35 && prob <= 0.65;
  const isLow = prob < 0.35;

  const badgeClass = isLow ? 'low' : isModerate ? 'moderate' : 'high';
  const badgeLabel = isLow
    ? 'Low Cardiac Disease Risk'
    : isModerate
    ? 'Moderate Cardiac Disease Risk'
    : 'High Cardiac Disease Risk';

  const gaugeColor = isLow ? 'var(--risk-low)' : isModerate ? 'var(--risk-mod)' : 'var(--risk-high)';

  // Semicircle SVG calculations
  const radius = 90;
  const circumference = Math.PI * radius; // ~282.74
  const strokeDashoffset = circumference * (1 - prob);

  // Generate plain-language sentence
  const topFactors = result.top_contributing_features ? result.top_contributing_features.slice(0, 3) : [];
  const primaryDriver = topFactors.length > 0 ? topFactors[0].feature : 'clinical markers';
  
  let plainExplanation = '';
  if (isHigh) {
    plainExplanation = `This patient shows a high likelihood of significant coronary artery disease (${percentage}%), heavily influenced by abnormal ${primaryDriver.toUpperCase()} and exercise perfusion response. Immediate physician evaluation is recommended.`;
  } else if (isModerate) {
    plainExplanation = `This patient demonstrates intermediate cardiac risk (${percentage}%), warranting non-invasive follow-up imaging and risk factor modification, with ${primaryDriver.toUpperCase()} being the primary contributing factor.`;
  } else {
    plainExplanation = `This patient presents low cardiovascular risk (${percentage}%), with cardiac biomarkers and stress parameters closely aligning with healthy baseline ranges.`;
  }

  // Model Agreement Analysis
  const comp = result.comparison || {};
  const compKeys = Object.keys(comp);
  let agreementStatus = 'High Multi-Model Consensus';
  let agreementColor = 'var(--risk-low)';
  let agreementBg = 'var(--risk-low-bg)';

  if (compKeys.length > 1) {
    const labels = compKeys.map((k) => comp[k].risk_label);
    const majority = labels.filter((l) => l === result.risk_label).length;
    const ratio = majority / labels.length;
    if (ratio === 1.0) {
      agreementStatus = '100% Unanimous Agreement across Quantum & Classical Models';
    } else if (ratio >= 0.75) {
      agreementStatus = `Strong Agreement (${majority}/${labels.length} Models Concordant)`;
    } else {
      agreementStatus = `Partial Agreement (${majority}/${labels.length} Models) — Borderline Decision`;
      agreementColor = 'var(--risk-mod)';
      agreementBg = 'var(--risk-mod-bg)';
    }
  }

  const handlePrintPdf = () => {
    window.print();
  };

  return (
    <div className="results-card" id="results-section">
      {/* Risk Gauge & Badge Header */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <div className="gauge-container" role="img" aria-label={`Risk gauge showing ${percentage}%`}>
          <svg className="gauge-svg" viewBox="0 0 240 130">
            {/* Background Arc */}
            <path
              d="M 30,120 A 90,90 0 0,1 210,120"
              fill="none"
              stroke="var(--border-subtle)"
              strokeWidth="18"
              strokeLinecap="round"
            />
            {/* Animated Value Arc */}
            <path
              d="M 30,120 A 90,90 0 0,1 210,120"
              fill="none"
              stroke={gaugeColor}
              strokeWidth="18"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              style={{ transition: 'stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1)' }}
            />
          </svg>

          <div className="gauge-center-content">
            <span className="gauge-percentage" style={{ color: gaugeColor }}>
              {percentage}%
            </span>
            <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>
              Probability
            </span>
          </div>
        </div>

        <div className={`risk-badge ${badgeClass}`}>
          {isHigh ? <ShieldAlert size={16} /> : <ShieldCheck size={16} />}
          <span>{badgeLabel}</span>
        </div>

        {/* Prediction Uncertainty & Confidence Interval */}
        {result.confidence_interval && (
          <div
            id="confidence-range-badge"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.6rem',
              marginTop: '1rem',
              padding: '0.45rem 1.15rem',
              borderRadius: '9999px',
              backgroundColor: 'var(--primary-50)',
              border: '1.5px solid var(--primary-200)',
              boxShadow: 'var(--shadow-sm)',
            }}
          >
            <span style={{ fontSize: '0.82rem', color: 'var(--primary-900)', fontWeight: 600 }}>
              Confidence Range:
            </span>
            <span style={{ fontSize: '0.95rem', fontWeight: 800, color: 'var(--primary-700)' }}>
              {result.confidence_interval.range_display}
            </span>
            <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
              ({result.confidence_interval.method})
            </span>
          </div>
        )}

        <p className="results-explanation" style={{ marginTop: '1.25rem' }}>
          {plainExplanation}
        </p>

        <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.75rem', fontSize: '0.8rem', color: 'var(--text-muted)', flexWrap: 'wrap', justifyContent: 'center' }}>
          <span>Primary Algorithm:</span>
          <strong style={{ color: 'var(--text-primary)' }}>{result.model_used}</strong>
          <span>•</span>
          <span>Inference Time:</span>
          <strong>{result.processing_time_ms} ms</strong>
          {result.brier_score !== undefined && result.brier_score !== null && (
            <>
              <span>•</span>
              <span>Brier Score:</span>
              <strong style={{ color: 'var(--text-primary)' }}>{result.brier_score.toFixed(4)}</strong>
            </>
          )}
        </div>
      </div>

      {/* Top 3 Contributing Factors Chart */}
      <div className="contributing-factors-card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
          <h3 className="factors-title" style={{ margin: 0 }}>
            <Layers size={18} color="var(--primary-600)" />
            <span>Top-3 Contributing Clinical Factors (PCA-Mapped)</span>
          </h3>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
            Permutation &times; PCA Loadings
          </span>
        </div>

        <div>
          {topFactors.map((factor, idx) => {
            const isElevating = factor.directional_impact.includes('Elevates');
            const percentWidth = Math.min(100, Math.max(15, Math.round(factor.importance * 180)));

            return (
              <div key={idx} className="factor-item">
                <div className="factor-header">
                  <div>
                    <span className="factor-name">{factor.feature.toUpperCase()}</span>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginLeft: '0.5rem' }}>
                      (Patient: <strong>{factor.patient_value}</strong> vs Avg: {factor.baseline_mean})
                    </span>
                  </div>
                  <span className={`factor-impact-badge ${isElevating ? 'elevates' : 'reduces'}`}>
                    {factor.directional_impact}
                  </span>
                </div>

                <div className="factor-bar-track">
                  <div
                    className="factor-bar-fill"
                    style={{
                      width: `${percentWidth}%`,
                      backgroundColor: isElevating ? 'var(--risk-high)' : 'var(--risk-low)',
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Quantum vs Classical Side-by-Side Comparison */}
      {result.comparison && (
        <div className="comparison-card">
          <div className="comparison-header">
            <div>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Cpu size={18} color="#8b5cf6" />
                <span>Quantum &amp; Classical Multi-Model Comparison</span>
              </h3>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                All models evaluated identically on patient&apos;s reduced quantum representation
              </p>
            </div>

            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.4rem',
                padding: '0.35rem 0.8rem',
                borderRadius: '9999px',
                fontSize: '0.8rem',
                fontWeight: 700,
                backgroundColor: agreementBg,
                color: agreementColor,
                border: `1px solid ${agreementColor}`,
              }}
            >
              <span>{agreementStatus}</span>
            </div>
          </div>

          <div className="comparison-grid">
            {Object.entries(result.comparison).map(([mName, mPred]) => {
              const isQuantumModel = mPred.model_type === 'Quantum';
              const mProb = Math.round(mPred.risk_probability * 100);
              const mColor = mPred.risk_level === 'High' ? 'var(--risk-high)' : mPred.risk_level === 'Moderate' ? 'var(--risk-mod)' : 'var(--risk-low)';

              return (
                <div key={mName} className={`model-pill-card ${isQuantumModel ? 'quantum' : ''}`}>
                  <div className="model-pill-title">
                    <span>{mName}</span>
                    <span
                      style={{
                        fontSize: '0.7rem',
                        padding: '0.15rem 0.45rem',
                        borderRadius: '9999px',
                        backgroundColor: isQuantumModel ? 'var(--quantum-bg)' : 'var(--bg-secondary)',
                        color: isQuantumModel ? 'var(--quantum-accent)' : 'var(--text-muted)',
                        fontWeight: 700,
                      }}
                    >
                      {mPred.model_type}
                    </span>
                  </div>

                  <div className="model-pill-prob" style={{ color: mColor }}>
                    {mProb}%
                  </div>

                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
                    <span>Stratification:</span>
                    <strong style={{ color: mColor }}>{mPred.risk_level}</strong>
                  </div>

                  {mPred.confidence_interval && (
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.35rem', paddingTop: '0.35rem', borderTop: '1px dashed var(--border-subtle)' }}>
                      <span>Confidence: </span>
                      <strong style={{ color: 'var(--text-primary)' }}>{mPred.confidence_interval.range_display}</strong>
                    </div>
                  )}

                  {mPred.brier_score !== undefined && mPred.brier_score !== null && (
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between', marginTop: '0.2rem' }}>
                      <span>Brier Score:</span>
                      <strong>{mPred.brier_score.toFixed(4)}</strong>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {result.quantum_representation && (
            <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'var(--bg-card)', borderRadius: 'var(--radius-sm)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Encoded Quantum Angles [0, &pi;]: </span>
              <strong>{result.quantum_representation.join(', ')}</strong>
            </div>
          )}
        </div>
      )}

      {/* Action Buttons: PDF Download, Edit, Reset */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button type="button" className="btn-secondary" onClick={onEdit}>
            <span>Edit Patient Vitals</span>
          </button>
          <button type="button" className="btn-secondary" onClick={onReset}>
            <RotateCcw size={16} />
            <span>New Assessment</span>
          </button>
        </div>

        <button type="button" className="btn-primary" onClick={handlePrintPdf}>
          <Printer size={16} />
          <span>Download PDF Report</span>
        </button>
      </div>

      {/* Printable Clinical PDF Report Summary (Visible when printing or previewing) */}
      <div className="print-report-summary" style={{ display: 'none' }}>
        <h2>CardioQuantum Clinical Risk Assessment Report</h2>
        <p>Date: {new Date().toLocaleDateString()} | Time: {new Date().toLocaleTimeString()}</p>
        <p><strong>Predicted Risk:</strong> {percentage}% ({badgeLabel})</p>
        <p><strong>Algorithm Used:</strong> {result.model_used} ({result.model_type})</p>
        <p><strong>Clinical Inputs:</strong> Age {patientData.age}, Sex {patientData.sex === 1 ? 'Male' : 'Female'}, BP {patientData.trestbps} mm Hg, Chol {patientData.chol} mg/dl, Max HR {patientData.thalach} bpm, ST Depression {patientData.oldpeak} mm, Blocked Vessels {patientData.ca}</p>
        <p><strong>Medical Notice:</strong> {result.medical_disclaimer}</p>
      </div>

      {/* Mandatory Medical Disclaimer Banner */}
      <div className="disclaimer-banner" role="note">
        <ShieldAlert size={20} style={{ flexShrink: 0 }} />
        <div>
          <strong>Medical Research Disclaimer:</strong> {result.medical_disclaimer}
        </div>
      </div>
    </div>
  );
}

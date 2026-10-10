import React from 'react';
import { ArrowRight, Sparkles, Cpu, HeartPulse } from 'lucide-react';

export default function Hero({ onStartClick }) {
  return (
    <section className="hero-section" aria-labelledby="hero-heading">
      <div className="hero-badge">
        <Sparkles size={14} />
        <span>Hybrid Quantum Machine Learning • 4-Qubit Variational Classifier</span>
      </div>

      <h1 id="hero-heading" className="hero-title">
        CardioQuantum Health Risk System
      </h1>

      <p className="hero-subtitle">
        Fast, interpretable cardiac disease risk assessment powered by Variational Quantum
        Circuits (PennyLane) and verified against state-of-the-art classical clinical benchmarks.
      </p>

      <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', flexWrap: 'wrap' }}>
        <button
          type="button"
          className="hero-cta-btn"
          onClick={onStartClick}
          aria-label="Start patient risk assessment"
        >
          <span>Start Assessment</span>
          <ArrowRight size={18} />
        </button>
      </div>
    </section>
  );
}

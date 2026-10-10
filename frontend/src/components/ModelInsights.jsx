import React, { useEffect, useState } from 'react';
import {
  Cpu,
  BarChart3,
  GitBranch,
  Info,
  CheckCircle2,
  ExternalLink,
  Layers,
  Sparkles,
} from 'lucide-react';

export default function ModelInsights() {
  const [metrics, setMetrics] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const [metricsRes, infoRes] = await Promise.all([
          fetch('/metrics'),
          fetch('/model-info'),
        ]);

        if (!metricsRes.ok || !infoRes.ok) {
          throw new Error('Failed to retrieve benchmark specifications from API server.');
        }

        const metricsData = await metricsRes.json();
        const infoData = await infoRes.json();
        setMetrics(metricsData);
        setModelInfo(infoData);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="insights-section">
        <div className="skeleton" style={{ height: '300px', width: '100%', borderRadius: 'var(--radius-xl)' }} />
        <div className="skeleton" style={{ height: '400px', width: '100%', borderRadius: 'var(--radius-xl)' }} />
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-banner">
        <span>Unable to load model insights: {error}</span>
      </div>
    );
  }

  return (
    <div className="insights-section" role="region" aria-label="Model Architecture & Performance Insights">
      {/* 3-Step "How the Quantum Model Works" Explainer */}
      <div className="wizard-card">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1.25rem' }}>
          <Sparkles size={22} color="var(--primary-600)" />
          <h2 style={{ fontSize: '1.35rem' }}>How the Variational Quantum Classifier (VQC) Works</h2>
        </div>

        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.75rem', lineHeight: 1.6 }}>
          CardioQuantum uses PennyLane to map classical clinical health markers into an exponentially large
          Hilbert state space, executing parameterized quantum rotations and entangled CNOT operations to determine cardiac disease probability.
        </p>

        <div className="quantum-explainer-grid">
          <div className="explainer-step">
            <div className="explainer-step-num">1</div>
            <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>Angle Feature Encoding</h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Patient features are projected to principal components and scaled strictly to [0, &pi;].
              Single-qubit R_Y(&phi;) rotations encode each clinical continuous value directly into quantum superposition states.
            </p>
          </div>

          <div className="explainer-step">
            <div className="explainer-step-num">2</div>
            <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>Entangled Ansatz Circuit</h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              A 2-layer <strong style={{ color: 'var(--primary-700)' }}>StronglyEntanglingLayers</strong> ansatz applies parameterized unitary rotations R(&alpha;, &beta;, &gamma;) paired with cyclic CNOT gates to induce non-local multi-qubit correlations.
            </p>
          </div>

          <div className="explainer-step">
            <div className="explainer-step-num">3</div>
            <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>Hermitian Measurement</h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              The expectation value &lang;Z&#8320;&rang; &isin; [-1, 1] on qubit 0 is measured and mapped to probability via P = (1 + &lang;Z&#8320;&rang;) / 2, optimized with Adam mini-batch backpropagation.
            </p>
          </div>
        </div>
      </div>

      {/* Cross-Algorithm Benchmark Comparison Table */}
      <div className="metrics-table-card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <BarChart3 size={20} color="var(--primary-600)" />
              <span>Cross-Algorithm Clinical Benchmark (5-Seed Average)</span>
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Evaluated strictly on identical test patients under zero-test-leakage conditions
            </p>
          </div>
          <span style={{ fontSize: '0.75rem', padding: '0.3rem 0.7rem', borderRadius: '9999px', background: 'var(--primary-50)', color: 'var(--primary-700)', fontWeight: 700 }}>
            Winner: {metrics?.production_model_winner}
          </span>
        </div>

        <table className="data-table">
          <thead>
            <tr>
              <th>Model Name</th>
              <th>Architecture</th>
              <th>Recall (Sensitivity)</th>
              <th>F1-Score</th>
              <th>ROC-AUC</th>
              <th>Accuracy</th>
              <th>Brier Score</th>
              <th>Parameters</th>
              <th>Train Time</th>
            </tr>
          </thead>
          <tbody>
            {metrics?.comparison_table?.map((row) => {
              const isWinner = row.model_name === metrics.production_model_winner;
              const brierVal = row.brier_score !== null && row.brier_score !== undefined
                ? row.brier_score
                : metrics?.brier_scores?.[row.model_name];

              return (
                <tr key={row.model_name} style={isWinner ? { backgroundColor: 'var(--primary-50)', fontWeight: 600 } : {}}>
                  <td>
                    <strong>{row.model_name}</strong> {isWinner && '⭐'}
                  </td>
                  <td>
                    <span style={{ fontSize: '0.75rem', padding: '0.2rem 0.5rem', borderRadius: '4px', background: row.model_type === 'Quantum' ? 'var(--quantum-bg)' : 'var(--bg-secondary)', color: row.model_type === 'Quantum' ? 'var(--quantum-accent)' : 'var(--text-muted)', fontWeight: 700 }}>
                      {row.model_type}
                    </span>
                  </td>
                  <td>{(row.recall * 100).toFixed(1)}%</td>
                  <td>{(row.f1_score * 100).toFixed(1)}%</td>
                  <td>{(row.roc_auc * 100).toFixed(1)}%</td>
                  <td>{(row.accuracy * 100).toFixed(1)}%</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>
                    {brierVal !== undefined && brierVal !== null ? Number(brierVal).toFixed(4) : '—'}
                  </td>
                  <td>{row.parameters}</td>
                  <td>{row.training_time_seconds.toFixed(3)}s</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Global Feature Importance Mapped Through PCA Loadings */}
      {modelInfo?.global_feature_importances && Object.keys(modelInfo.global_feature_importances).length > 0 && (
        <div className="wizard-card" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
            <div>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Layers size={20} color="var(--primary-600)" />
                <span>Global Clinical Feature Importance (PCA-Mapped)</span>
              </h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Permutation importance evaluated on the 4 reduced quantum features, mapped linearly back to original biomarkers through PCA eigenvectors (V matrix)
              </p>
            </div>
            <span style={{ fontSize: '0.75rem', padding: '0.25rem 0.65rem', borderRadius: '9999px', background: 'var(--bg-secondary)', color: 'var(--text-secondary)', fontWeight: 600 }}>
              13 Clinical Biomarkers
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '0.85rem' }}>
            {Object.entries(modelInfo.global_feature_importances)
              .sort((a, b) => b[1] - a[1])
              .map(([feat, imp]) => {
                const pct = Math.round(imp * 100);
                return (
                  <div
                    key={feat}
                    style={{
                      padding: '0.65rem 0.85rem',
                      background: 'var(--bg-card-subtle)',
                      borderRadius: 'var(--radius-sm)',
                      border: '1px solid var(--border-subtle)',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '0.35rem' }}>
                      <strong style={{ color: 'var(--text-primary)', textTransform: 'uppercase' }}>{feat}</strong>
                      <span style={{ color: 'var(--primary-700)', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
                        {(imp * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{ height: '6px', background: 'var(--border-subtle)', borderRadius: '9999px', overflow: 'hidden' }}>
                      <div
                        style={{
                          height: '100%',
                          width: `${Math.min(100, pct * 4)}%`,
                          background: 'linear-gradient(90deg, var(--primary-500), var(--primary-700))',
                          borderRadius: '9999px',
                        }}
                      />
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {/* Visual Research Diagrams: Circuit, ROC Overlay, & Calibration Curves */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem' }}>
        {/* Quantum Circuit Diagram */}
        <div className="wizard-card" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Cpu size={18} color="var(--primary-600)" />
            <span>4-Qubit Variational Quantum Circuit</span>
          </h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
            PennyLane <code>qml.draw_mpl</code> showing AngleEmbedding &amp; StronglyEntanglingLayers.
          </p>
          <div style={{ background: '#ffffff', padding: '0.5rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'center' }}>
            <img
              src="/static/vqc_circuit.png"
              alt="4-Qubit VQC Quantum Circuit Architecture Diagram"
              style={{ maxWidth: '100%', height: 'auto', borderRadius: 'var(--radius-sm)' }}
              loading="lazy"
            />
          </div>
        </div>

        {/* ROC Curves Overlay */}
        <div className="wizard-card" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <BarChart3 size={18} color="var(--primary-600)" />
            <span>Multi-Model ROC Curves Overlay</span>
          </h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
            Receiver Operating Characteristic overlay across all 6 models on UCI Cleveland test split.
          </p>
          <div style={{ background: '#ffffff', padding: '0.5rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'center' }}>
            <img
              src="/static/figures/roc_curves_overlay.png"
              alt="Multi-Model ROC Curves Overlay"
              style={{ maxWidth: '100%', height: 'auto', borderRadius: 'var(--radius-sm)' }}
              loading="lazy"
            />
          </div>
        </div>

        {/* Probability Calibration Curves & Brier Scores */}
        <div className="wizard-card" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <GitBranch size={18} color="var(--primary-600)" />
            <span>Probability Calibration &amp; Brier Scores</span>
          </h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
            Reliability diagrams comparing predicted probabilities against true observed clinical risk.
          </p>
          <div style={{ background: '#ffffff', padding: '0.5rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'center' }}>
            <img
              src="/static/figures/calibration_curves.png"
              alt="Probability Calibration Curves and Reliability Diagram"
              style={{ maxWidth: '100%', height: 'auto', borderRadius: 'var(--radius-sm)' }}
              loading="lazy"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

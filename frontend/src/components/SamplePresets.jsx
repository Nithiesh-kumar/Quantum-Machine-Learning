import React from 'react';
import { UserCheck, Check, Clock } from 'lucide-react';

export const PATIENT_PRESETS = {
  low: {
    id: 'low',
    name: 'Low Risk Patient',
    description: '42yo Female, normal vitals, 0 vessels blocked',
    color: 'var(--risk-low)',
    data: {
      age: 42.0,
      sex: 0,
      cp: 2,
      trestbps: 118.0,
      chol: 195.0,
      fbs: 0,
      restecg: 0,
      thalach: 175.0,
      exang: 0,
      oldpeak: 0.0,
      slope: 1,
      ca: 0.0,
      thal: 3.0,
    },
  },
  moderate: {
    id: 'moderate',
    name: 'Moderate Risk Patient',
    description: '54yo Male, atypical angina, mild oldpeak',
    color: 'var(--risk-mod)',
    data: {
      age: 54.0,
      sex: 1,
      cp: 3,
      trestbps: 135.0,
      chol: 230.0,
      fbs: 0,
      restecg: 1,
      thalach: 145.0,
      exang: 0,
      oldpeak: 1.2,
      slope: 2,
      ca: 1.0,
      thal: 6.0,
    },
  },
  high: {
    id: 'high',
    name: 'High Risk Patient',
    description: '63yo Male, asymptomatic, 2 vessels blocked, reversible defect',
    color: 'var(--risk-high)',
    data: {
      age: 63.0,
      sex: 1,
      cp: 4,
      trestbps: 152.0,
      chol: 275.0,
      fbs: 1,
      restecg: 2,
      thalach: 128.0,
      exang: 1,
      oldpeak: 2.8,
      slope: 2,
      ca: 2.0,
      thal: 7.0,
    },
  },
};

export default function SamplePresets({ activePreset, onSelectPreset, lastSaved }) {
  return (
    <div className="presets-card" role="region" aria-label="Sample Patient Presets">
      <div className="presets-label">
        <UserCheck size={18} color="var(--primary-600)" />
        <span>Load Sample Patient Presets:</span>
      </div>

      <div className="presets-group">
        {Object.values(PATIENT_PRESETS).map((preset) => {
          const isActive = activePreset === preset.id;
          return (
            <button
              key={preset.id}
              type="button"
              className={`preset-pill ${isActive ? 'active' : ''}`}
              onClick={() => onSelectPreset(preset)}
              aria-label={`Load ${preset.name} (${preset.description})`}
              title={preset.description}
            >
              <span
                style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  backgroundColor: preset.color,
                }}
              />
              <span>{preset.name}</span>
              {isActive && <Check size={14} />}
            </button>
          );
        })}
      </div>

      {lastSaved && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          <Clock size={12} />
          <span>Autosaved {lastSaved}</span>
        </div>
      )}
    </div>
  );
}

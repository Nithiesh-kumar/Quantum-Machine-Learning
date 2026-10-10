import React, { useState } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  HelpCircle,
  Sparkles,
  Heart,
  Stethoscope,
  FlaskConical,
  Zap,
} from 'lucide-react';

export default function Wizard({
  patientData,
  onChange,
  onReset,
  onSubmit,
  loading,
  currentStep,
  setCurrentStep,
}) {
  const [errors, setErrors] = useState({});
  const [compareModels, setCompareModels] = useState(true);
  const [selectedModel, setSelectedModel] = useState('production');

  // Inline Validation Rulebook
  const validateStep = (step) => {
    const errs = {};
    if (step === 1) {
      if (!patientData.age || patientData.age < 18 || patientData.age > 120) {
        errs.age = 'Age must be between 18 and 120 years.';
      }
    } else if (step === 2) {
      if (!patientData.trestbps || patientData.trestbps < 60 || patientData.trestbps > 260) {
        errs.trestbps = 'Resting BP must be between 60 and 260 mm Hg.';
      }
      if (!patientData.thalach || patientData.thalach < 40 || patientData.thalach > 250) {
        errs.thalach = 'Max heart rate must be between 40 and 250 bpm.';
      }
    } else if (step === 3) {
      if (!patientData.chol || patientData.chol < 80 || patientData.chol > 650) {
        errs.chol = 'Cholesterol must be between 80 and 650 mg/dl.';
      }
      if (patientData.oldpeak === undefined || patientData.oldpeak < 0 || patientData.oldpeak > 8) {
        errs.oldpeak = 'ST depression must be between 0.0 and 8.0 mm.';
      }
    }
    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleNext = () => {
    if (validateStep(currentStep)) {
      setCurrentStep((prev) => Math.min(prev + 1, 3));
    }
  };

  const handleBack = () => {
    setCurrentStep((prev) => Math.max(prev - 1, 1));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validateStep(currentStep)) {
      onSubmit({ model_name: selectedModel, compare: compareModels });
    }
  };

  const updateField = (field, value) => {
    onChange(field, value);
    if (errors[field]) {
      setErrors((prev) => {
        const updated = { ...prev };
        delete updated[field];
        return updated;
      });
    }
  };

  return (
    <div className="wizard-card" id="wizard-form">
      {/* Step Progress Indicator */}
      <div className="stepper-header" role="navigation" aria-label="Assessment Progress">
        <div className="stepper-progress-line" aria-hidden="true">
          <div
            className="stepper-progress-fill"
            style={{ width: currentStep === 1 ? '0%' : currentStep === 2 ? '50%' : '100%' }}
          />
        </div>

        <div
          className={`step-indicator ${currentStep === 1 ? 'active' : ''} ${currentStep > 1 ? 'completed' : ''}`}
          onClick={() => currentStep > 1 && setCurrentStep(1)}
          role="button"
          tabIndex={0}
          aria-label="Step 1: Patient Demographics & Symptoms"
        >
          <div className="step-circle">{currentStep > 1 ? <CheckCircle2 size={20} /> : '1'}</div>
          <span className="step-name">1. Demographics</span>
        </div>

        <div
          className={`step-indicator ${currentStep === 2 ? 'active' : ''} ${currentStep > 2 ? 'completed' : ''}`}
          onClick={() => currentStep > 2 && setCurrentStep(2)}
          role="button"
          tabIndex={0}
          aria-label="Step 2: Cardiovascular Vitals"
        >
          <div className="step-circle">{currentStep > 2 ? <CheckCircle2 size={20} /> : '2'}</div>
          <span className="step-name">2. Vitals & ECG</span>
        </div>

        <div
          className={`step-indicator ${currentStep === 3 ? 'active' : ''}`}
          onClick={() => {}}
          role="button"
          tabIndex={0}
          aria-label="Step 3: Lab Results & Imaging"
        >
          <div className="step-circle">3</div>
          <span className="step-name">3. Lab & Imaging</span>
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        {/* ================================================================= */}
        {/* STEP 1: DEMOGRAPHICS & CHEST SYMPTOMS */}
        {/* ================================================================= */}
        {currentStep === 1 && (
          <div className="step-content">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1.5rem' }}>
              <Heart size={22} color="var(--primary-600)" />
              <h2 style={{ fontSize: '1.25rem' }}>Step 1: Patient Demographics & Chest Symptoms</h2>
            </div>

            <div className="form-grid">
              {/* Age */}
              <div className="form-group">
                <div className="field-label-row">
                  <label htmlFor="field-age" className="field-label">
                    Patient Age
                    <span className="field-tooltip" title="Patient age in completed years (18–120)">
                      <HelpCircle size={14} />
                    </span>
                  </label>
                  <span className="field-normal-hint">Normal: 18 – 65 yrs</span>
                </div>
                <input
                  id="field-age"
                  type="number"
                  step="1"
                  min="18"
                  max="120"
                  className={`text-input ${errors.age ? 'error' : ''}`}
                  value={patientData.age}
                  onChange={(e) => updateField('age', parseFloat(e.target.value) || '')}
                  placeholder="e.g. 58"
                  required
                />
                {errors.age && <span className="error-message">{errors.age}</span>}
              </div>

              {/* Sex */}
              <div className="form-group">
                <div className="field-label-row">
                  <span className="field-label">Biological Sex</span>
                  <span className="field-normal-hint">Clinical standard</span>
                </div>
                <div className="segmented-group" role="radiogroup" aria-label="Biological Sex">
                  <button
                    type="button"
                    className={`segmented-btn ${patientData.sex === 0 ? 'active' : ''}`}
                    onClick={() => updateField('sex', 0)}
                    role="radio"
                    aria-checked={patientData.sex === 0}
                  >
                    Female (0)
                  </button>
                  <button
                    type="button"
                    className={`segmented-btn ${patientData.sex === 1 ? 'active' : ''}`}
                    onClick={() => updateField('sex', 1)}
                    role="radio"
                    aria-checked={patientData.sex === 1}
                  >
                    Male (1)
                  </button>
                </div>
              </div>

              {/* Chest Pain Type */}
              <div className="form-group full-width">
                <div className="field-label-row">
                  <span className="field-label">
                    Chest Pain Sensation (Angina Type)
                    <span className="field-tooltip" title="Classified per Cleveland Clinic criteria">
                      <HelpCircle size={14} />
                    </span>
                  </span>
                  <span className="field-normal-hint">Select matching symptom</span>
                </div>
                <div className="card-options-grid" role="radiogroup" aria-label="Chest Pain Sensation">
                  {[
                    { val: 1, title: 'Typical Angina', desc: 'Substernal pressure with exercise or stress' },
                    { val: 2, title: 'Atypical Angina', desc: 'Chest discomfort lacking classic patterns' },
                    { val: 3, title: 'Non-Anginal Pain', desc: 'Sharp, localized, or musculoskeletal ache' },
                    { val: 4, title: 'Asymptomatic', desc: 'Silent ischemia / no overt chest pain' },
                  ].map((opt) => (
                    <div
                      key={opt.val}
                      className={`card-option ${patientData.cp === opt.val ? 'active' : ''}`}
                      onClick={() => updateField('cp', opt.val)}
                      role="radio"
                      aria-checked={patientData.cp === opt.val}
                      tabIndex={0}
                    >
                      <span className="card-option-title">{opt.title}</span>
                      <span className="card-option-desc">{opt.desc}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Exercise Induced Angina */}
              <div className="form-group full-width">
                <div className="field-label-row">
                  <span className="field-label">
                    Does Physical Exercise Trigger Chest Angina?
                    <span className="field-tooltip" title="Angina elicited during treadmill exercise test">
                      <HelpCircle size={14} />
                    </span>
                  </span>
                  <span className="field-normal-hint">Normal: No</span>
                </div>
                <div className="segmented-group" role="radiogroup" aria-label="Exercise Induced Angina">
                  <button
                    type="button"
                    className={`segmented-btn ${patientData.exang === 0 ? 'active' : ''}`}
                    onClick={() => updateField('exang', 0)}
                    role="radio"
                    aria-checked={patientData.exang === 0}
                  >
                    No Exercise Angina (0)
                  </button>
                  <button
                    type="button"
                    className={`segmented-btn ${patientData.exang === 1 ? 'active' : ''}`}
                    onClick={() => updateField('exang', 1)}
                    role="radio"
                    aria-checked={patientData.exang === 1}
                  >
                    Yes, Triggers Angina (1)
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ================================================================= */}
        {/* STEP 2: CARDIOVASCULAR VITALS & ECG */}
        {/* ================================================================= */}
        {currentStep === 2 && (
          <div className="step-content">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1.5rem' }}>
              <Stethoscope size={22} color="var(--primary-600)" />
              <h2 style={{ fontSize: '1.25rem' }}>Step 2: Cardiovascular Vitals & ECG</h2>
            </div>

            <div className="form-grid">
              {/* Resting Blood Pressure */}
              <div className="form-group">
                <div className="field-label-row">
                  <label htmlFor="field-trestbps" className="field-label">
                    Resting Blood Pressure
                    <span className="field-tooltip" title="Measured upon quiet hospital admission">
                      <HelpCircle size={14} />
                    </span>
                  </label>
                  <span className="field-normal-hint">Normal: 90 – 120 mm Hg</span>
                </div>
                <input
                  id="field-trestbps"
                  type="number"
                  step="1"
                  min="60"
                  max="260"
                  className={`text-input ${errors.trestbps ? 'error' : ''}`}
                  value={patientData.trestbps}
                  onChange={(e) => updateField('trestbps', parseFloat(e.target.value) || '')}
                  placeholder="e.g. 130"
                  required
                />
                {errors.trestbps && <span className="error-message">{errors.trestbps}</span>}
              </div>

              {/* Max Heart Rate */}
              <div className="form-group">
                <div className="field-label-row">
                  <label htmlFor="field-thalach" className="field-label">
                    Maximum Heart Rate Achieved
                    <span className="field-tooltip" title="Peak heart rate reached during stress test (bpm)">
                      <HelpCircle size={14} />
                    </span>
                  </label>
                  <span className="field-normal-hint">Normal: 120 – 180 bpm</span>
                </div>
                <input
                  id="field-thalach"
                  type="number"
                  step="1"
                  min="40"
                  max="250"
                  className={`text-input ${errors.thalach ? 'error' : ''}`}
                  value={patientData.thalach}
                  onChange={(e) => updateField('thalach', parseFloat(e.target.value) || '')}
                  placeholder="e.g. 150"
                  required
                />
                {errors.thalach && <span className="error-message">{errors.thalach}</span>}
              </div>

              {/* Fasting Blood Sugar */}
              <div className="form-group">
                <div className="field-label-row">
                  <span className="field-label">Fasting Blood Sugar &gt; 120 mg/dl</span>
                  <span className="field-normal-hint">Normal: &lt; 120 mg/dl</span>
                </div>
                <div className="segmented-group" role="radiogroup" aria-label="Fasting Blood Sugar">
                  <button
                    type="button"
                    className={`segmented-btn ${patientData.fbs === 0 ? 'active' : ''}`}
                    onClick={() => updateField('fbs', 0)}
                    role="radio"
                    aria-checked={patientData.fbs === 0}
                  >
                    Normal (&le; 120)
                  </button>
                  <button
                    type="button"
                    className={`segmented-btn ${patientData.fbs === 1 ? 'active' : ''}`}
                    onClick={() => updateField('fbs', 1)}
                    role="radio"
                    aria-checked={patientData.fbs === 1}
                  >
                    Elevated (&gt; 120)
                  </button>
                </div>
              </div>

              {/* Resting ECG */}
              <div className="form-group full-width">
                <div className="field-label-row">
                  <span className="field-label">
                    Resting Electrocardiogram (ECG) Finding
                    <span className="field-tooltip" title="Resting electrical baseline recording">
                      <HelpCircle size={14} />
                    </span>
                  </span>
                  <span className="field-normal-hint">Normal: Normal rhythm</span>
                </div>
                <div className="card-options-grid" role="radiogroup" aria-label="Resting Electrocardiogram">
                  {[
                    { val: 0, title: 'Normal Rhythm (0)', desc: 'No significant repolarization anomalies' },
                    { val: 1, title: 'ST-T Wave Abnormality (1)', desc: 'T wave inversions or ST depression > 0.05 mV' },
                    { val: 2, title: 'Left Ventricular Hypertrophy (2)', desc: 'Definite LVH by Estes voltage criteria' },
                  ].map((opt) => (
                    <div
                      key={opt.val}
                      className={`card-option ${patientData.restecg === opt.val ? 'active' : ''}`}
                      onClick={() => updateField('restecg', opt.val)}
                      role="radio"
                      aria-checked={patientData.restecg === opt.val}
                      tabIndex={0}
                    >
                      <span className="card-option-title">{opt.title}</span>
                      <span className="card-option-desc">{opt.desc}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ================================================================= */}
        {/* STEP 3: LAB RESULTS & CARDIAC IMAGING */}
        {/* ================================================================= */}
        {currentStep === 3 && (
          <div className="step-content">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1.5rem' }}>
              <FlaskConical size={22} color="var(--primary-600)" />
              <h2 style={{ fontSize: '1.25rem' }}>Step 3: Laboratory & Advanced Cardiac Imaging</h2>
            </div>

            <div className="form-grid">
              {/* Serum Cholesterol */}
              <div className="form-group">
                <div className="field-label-row">
                  <label htmlFor="field-chol" className="field-label">
                    Serum Total Cholesterol
                    <span className="field-tooltip" title="Total serum cholesterol (mg/dl)">
                      <HelpCircle size={14} />
                    </span>
                  </label>
                  <span className="field-normal-hint">Normal: 125 – 200 mg/dl</span>
                </div>
                <input
                  id="field-chol"
                  type="number"
                  step="1"
                  min="80"
                  max="650"
                  className={`text-input ${errors.chol ? 'error' : ''}`}
                  value={patientData.chol}
                  onChange={(e) => updateField('chol', parseFloat(e.target.value) || '')}
                  placeholder="e.g. 240"
                  required
                />
                {errors.chol && <span className="error-message">{errors.chol}</span>}
              </div>

              {/* ST Depression (oldpeak) */}
              <div className="form-group">
                <div className="field-label-row">
                  <label htmlFor="field-oldpeak" className="field-label">
                    Exercise ST Depression (Oldpeak)
                    <span className="field-tooltip" title="ST depression induced by exercise relative to rest (mm)">
                      <HelpCircle size={14} />
                    </span>
                  </label>
                  <span className="field-normal-hint">Normal: 0.0 – 1.0 mm</span>
                </div>
                <input
                  id="field-oldpeak"
                  type="number"
                  step="0.1"
                  min="0.0"
                  max="8.0"
                  className={`text-input ${errors.oldpeak ? 'error' : ''}`}
                  value={patientData.oldpeak}
                  onChange={(e) => updateField('oldpeak', parseFloat(e.target.value) || 0.0)}
                  placeholder="e.g. 1.8"
                  required
                />
                {errors.oldpeak && <span className="error-message">{errors.oldpeak}</span>}
              </div>

              {/* Peak ST Slope */}
              <div className="form-group">
                <div className="field-label-row">
                  <span className="field-label">Peak ST Slope</span>
                  <span className="field-normal-hint">Exercise response</span>
                </div>
                <div className="segmented-group" role="radiogroup" aria-label="Peak ST Slope">
                  {[
                    { val: 1, label: 'Upsloping (1)' },
                    { val: 2, label: 'Flat (2)' },
                    { val: 3, label: 'Downsloping (3)' },
                  ].map((s) => (
                    <button
                      key={s.val}
                      type="button"
                      className={`segmented-btn ${patientData.slope === s.val ? 'active' : ''}`}
                      onClick={() => updateField('slope', s.val)}
                      role="radio"
                      aria-checked={patientData.slope === s.val}
                    >
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Major Vessels (Fluoroscopy) */}
              <div className="form-group">
                <div className="field-label-row">
                  <span className="field-label">
                    Major Vessels Blocked (Fluoroscopy)
                    <span className="field-tooltip" title="Number of major vessels (0–3) colored by flourosopy">
                      <HelpCircle size={14} />
                    </span>
                  </span>
                  <span className="field-normal-hint">Normal: 0 vessels</span>
                </div>
                <div className="segmented-group" role="radiogroup" aria-label="Major Vessels Blocked">
                  {[0, 1, 2, 3].map((num) => (
                    <button
                      key={num}
                      type="button"
                      className={`segmented-btn ${patientData.ca === num ? 'active' : ''}`}
                      onClick={() => updateField('ca', num)}
                      role="radio"
                      aria-checked={patientData.ca === num}
                    >
                      {num} {num === 1 ? 'vessel' : 'vessels'}
                    </button>
                  ))}
                </div>
              </div>

              {/* Thallium Stress Defect */}
              <div className="form-group full-width">
                <div className="field-label-row">
                  <span className="field-label">
                    Thallium Stress Perfusion Scintigraphy (Thal)
                    <span className="field-tooltip" title="Nuclear imaging perfusion status">
                      <HelpCircle size={14} />
                    </span>
                  </span>
                  <span className="field-normal-hint">Normal: Normal blood flow</span>
                </div>
                <div className="card-options-grid" role="radiogroup" aria-label="Thallium Stress Scintigraphy">
                  {[
                    { val: 3.0, title: 'Normal Perfusion (3)', desc: 'Uniform radioactive tracer uptake across myocardium' },
                    { val: 6.0, title: 'Fixed Defect (6)', desc: 'Permanent scar tissue from previous myocardial infarction' },
                    { val: 7.0, title: 'Reversible Defect (7)', desc: 'Active, reversible ischemia during exercise stress' },
                  ].map((opt) => (
                    <div
                      key={opt.val}
                      className={`card-option ${patientData.thal === opt.val ? 'active' : ''}`}
                      onClick={() => updateField('thal', opt.val)}
                      role="radio"
                      aria-checked={patientData.thal === opt.val}
                      tabIndex={0}
                    >
                      <span className="card-option-title">{opt.title}</span>
                      <span className="card-option-desc">{opt.desc}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Multi-Model Compare Checkbox & Model Selector */}
              <div className="form-group full-width" style={{ marginTop: '0.5rem', background: 'var(--bg-secondary)', padding: '1rem', borderRadius: 'var(--radius-md)' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.875rem', fontWeight: 600 }}>
                    <input
                      type="checkbox"
                      checked={compareModels}
                      onChange={(e) => setCompareModels(e.target.checked)}
                      style={{ width: '18px', height: '18px', accentColor: 'var(--primary-600)' }}
                    />
                    <span>Compare Quantum (VQC, QSVC) &amp; Classical Baselines Side-by-Side</span>
                  </label>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Primary:</span>
                    <select
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      style={{
                        padding: '0.35rem 0.65rem',
                        borderRadius: 'var(--radius-sm)',
                        border: '1px solid var(--border-subtle)',
                        background: 'var(--bg-card)',
                        color: 'var(--text-primary)',
                        fontSize: '0.8rem',
                        fontWeight: 600,
                      }}
                    >
                      <option value="production">Production Model (MLP)</option>
                      <option value="VQC">Variational Quantum (PennyLane)</option>
                      <option value="QSVC">Quantum Support Vector (Qiskit)</option>
                      <option value="Random Forest">Random Forest</option>
                      <option value="Logistic Regression">Logistic Regression</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Wizard Footer Controls */}
        <div className="wizard-footer">
          <div>
            {currentStep > 1 && (
              <button type="button" className="btn-secondary" onClick={handleBack} disabled={loading}>
                <ArrowLeft size={16} />
                <span>Previous Step</span>
              </button>
            )}
          </div>

          <div style={{ display: 'flex', gap: '0.75rem' }}>
            {currentStep < 3 ? (
              <button type="button" className="btn-primary" onClick={handleNext}>
                <span>Continue to {currentStep === 1 ? 'Vitals' : 'Lab Results'}</span>
                <ArrowRight size={16} />
              </button>
            ) : (
              <button type="submit" className="btn-primary" disabled={loading} style={{ minWidth: '180px' }}>
                {loading ? (
                  <>
                    <Zap size={18} className="spin" />
                    <span>Executing Quantum Circuit...</span>
                  </>
                ) : (
                  <>
                    <Sparkles size={18} />
                    <span>Run Quantum Assessment</span>
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </form>
    </div>
  );
}

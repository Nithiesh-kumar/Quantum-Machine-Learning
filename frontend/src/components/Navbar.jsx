import React from 'react';
import { Activity, Moon, Sun, ShieldCheck } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, theme, toggleTheme, apiOnline }) {
  return (
    <header className="navbar" role="banner">
      <div className="brand">
        <div className="brand-icon" aria-hidden="true">
          <Activity size={22} strokeWidth={2.5} />
        </div>
        <div>
          <span>CardioQuantum</span>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--primary-600)', marginLeft: '0.4rem' }}>
            QML CLINICAL
          </span>
        </div>
      </div>

      <nav className="nav-actions" aria-label="Main Navigation">
        <button
          type="button"
          className={`tab-btn ${activeTab === 'assessment' ? 'active' : ''}`}
          onClick={() => setActiveTab('assessment')}
          aria-current={activeTab === 'assessment' ? 'page' : undefined}
        >
          Patient Assessment
        </button>
        <button
          type="button"
          className={`tab-btn ${activeTab === 'insights' ? 'active' : ''}`}
          onClick={() => setActiveTab('insights')}
          aria-current={activeTab === 'insights' ? 'page' : undefined}
        >
          Model Insights
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginLeft: '0.5rem' }}>
          <span
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.3rem',
              fontSize: '0.75rem',
              padding: '0.25rem 0.6rem',
              borderRadius: '9999px',
              backgroundColor: apiOnline ? 'var(--risk-low-bg)' : 'var(--risk-high-bg)',
              color: apiOnline ? 'var(--risk-low)' : 'var(--risk-high)',
              fontWeight: 700,
            }}
            title={apiOnline ? 'FastAPI Backend Online' : 'FastAPI Backend Offline'}
          >
            <span
              style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                backgroundColor: apiOnline ? 'var(--risk-low)' : 'var(--risk-high)',
              }}
            />
            {apiOnline ? 'API Ready' : 'API Offline'}
          </span>

          <button
            type="button"
            className="theme-toggle-btn"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === 'light' ? 'Dark' : 'Light'} Mode`}
            title={`Switch to ${theme === 'light' ? 'Dark' : 'Light'} Mode`}
          >
            {theme === 'light' ? <Moon size={18} /> : <Sun size={18} />}
          </button>
        </div>
      </nav>
    </header>
  );
}

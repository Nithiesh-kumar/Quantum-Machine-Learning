import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/datasets': 'http://127.0.0.1:8000',
      '/predict': 'http://127.0.0.1:8000',
      '/metrics': 'http://127.0.0.1:8000',
      '/model-info': 'http://127.0.0.1:8000',
      '/static': 'http://127.0.0.1:8000',
    },
  },
})

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

const proxyConfig = {
  '/api': {
    target: 'http://localhost:18003',
    changeOrigin: true,
  },
  '/health': {
    target: 'http://localhost:18003',
    changeOrigin: true,
  },
  '/ws': {
    target: 'ws://localhost:18003',
    ws: true,
  },
}

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: proxyConfig,
  },
  preview: {
    port: 3000,
    proxy: proxyConfig,
  },
})

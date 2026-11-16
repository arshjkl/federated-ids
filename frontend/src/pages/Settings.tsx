'use client'

import { Shield, Database, Server, Bell, Palette } from 'lucide-react'
import { useTheme } from '@/hooks/useTheme'

export function Settings() {
  const { theme, setTheme } = useTheme()

  return (
    <div className="space-y-6 animate-in max-w-3xl">
      <div>
        <h1 className="text-2xl font-bold text-dark-50">Settings</h1>
        <p className="text-dark-400 mt-1">Platform configuration</p>
      </div>

      <div className="p-6 rounded-xl border border-dark-700 bg-dark-900/50 space-y-6">
        <section>
          <h3 className="text-lg font-semibold text-dark-50 mb-4 flex items-center gap-2">
            <Palette className="w-5 h-5" />
            Appearance
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {['light', 'dark', 'system'].map((t) => (
              <button
                key={t}
                onClick={() => setTheme(t as 'light' | 'dark')}
                className={`p-4 rounded-lg border-2 text-left transition-all ${
                  theme === t
                    ? 'border-primary-500 bg-primary-500/10 text-primary-400'
                    : 'border-dark-700 bg-dark-800/50 text-dark-300 hover:border-dark-600'
                }`}
              >
                <p className="font-medium capitalize">{t}</p>
                <p className="text-sm text-dark-400 mt-1">
                  {t === 'system' ? 'Follow system preference' : `${t} mode`}
                </p>
              </button>
            ))}
          </div>
        </section>

        <section className="border-t border-dark-700 pt-6">
          <h3 className="text-lg font-semibold text-dark-50 mb-4 flex items-center gap-2">
            <Server className="w-5 h-5" />
            API Configuration
          </h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-dark-400 mb-1">Backend URL</label>
              <input
                type="text"
                value="http://localhost:8000"
                className="w-full px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 placeholder-dark-400 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                readOnly
              />
            </div>
            <div>
              <label className="block text-sm text-dark-400 mb-1">WebSocket URL</label>
              <input
                type="text"
                value="ws://localhost:8000"
                className="w-full px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 placeholder-dark-400 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                readOnly
              />
            </div>
          </div>
        </section>

        <section className="border-t border-dark-700 pt-6">
          <h3 className="text-lg font-semibold text-dark-50 mb-4 flex items-center gap-2">
            <Database className="w-5 h-5" />
            Database
          </h3>
          <p className="text-dark-400">PostgreSQL: postgresql+asyncpg://postgres:postgres@localhost:5432/federated_ids</p>
        </section>

        <section className="border-t border-dark-700 pt-6">
          <h3 className="text-lg font-semibold text-dark-50 mb-4 flex items-center gap-2">
            <Shield className="w-5 h-5" />
            Security
          </h3>
          <p className="text-dark-400">Authentication and authorization settings (to be implemented)</p>
        </section>

        <section className="border-t border-dark-700 pt-6">
          <h3 className="text-lg font-semibold text-dark-50 mb-4 flex items-center gap-2">
            <Bell className="w-5 h-5" />
            Notifications
          </h3>
          <p className="text-dark-400">Notification preferences (to be implemented)</p>
        </section>
      </div>
    </div>
  )
}
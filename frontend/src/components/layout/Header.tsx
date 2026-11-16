'use client'

import { useTheme } from '@/hooks/useTheme'
import { Sun, Moon, Bell } from 'lucide-react'
import { clsx } from 'clsx'

export function Header() {
  const { theme, toggleTheme } = useTheme()

  return (
    <header className="h-16 bg-dark-900/80 backdrop-blur-sm border-b border-dark-700 flex items-center justify-between px-6">
      <div className="flex items-center gap-4">
        <h2 className="text-lg font-semibold text-dark-50 hidden md:block">
          Federated Learning Intrusion Detection Platform
        </h2>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={toggleTheme}
          className={clsx(
            'p-2 rounded-lg transition-colors',
            theme === 'dark'
              ? 'text-yellow-400 hover:bg-dark-800'
              : 'text-orange-500 hover:bg-dark-100'
          )}
          aria-label="Toggle theme"
        >
          {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
        </button>

        <button className="p-2 rounded-lg text-dark-400 hover:bg-dark-800 hover:text-dark-50 transition-colors">
          <Bell className="w-5 h-5" />
        </button>

        <div className="w-8 h-8 rounded-full bg-primary-500 flex items-center justify-center text-dark-950 font-medium">
          A
        </div>
      </div>
    </header>
  )
}
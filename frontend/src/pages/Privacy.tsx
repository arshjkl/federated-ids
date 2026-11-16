'use client'

export function Privacy() {
  return (
    <div className="space-y-6 animate-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-50">Privacy Analytics</h1>
          <p className="text-dark-400 mt-1">Differential privacy metrics and compliance</p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className="p-4 rounded-xl border border-dark-700 bg-dark-900/50">
          <p className="text-sm text-dark-400">Total Epsilon (ε)</p>
          <p className="text-2xl font-bold text-dark-50 mt-1">0.000</p>
        </div>
        <div className="p-4 rounded-xl border border-dark-700 bg-dark-900/50">
          <p className="text-sm text-dark-400">Delta (δ)</p>
          <p className="text-2xl font-bold text-dark-50 mt-1">0</p>
        </div>
        <div className="p-4 rounded-xl border border-dark-700 bg-dark-900/50">
          <p className="text-sm text-dark-400">Noise Multiplier</p>
          <p className="text-2xl font-bold text-dark-50 mt-1">0.00</p>
        </div>
        <div className="p-4 rounded-xl border border-dark-700 bg-dark-900/50">
          <p className="text-sm text-dark-400">Max Grad Norm</p>
          <p className="text-2xl font-bold text-dark-50 mt-1">0.00</p>
        </div>
      </div>

      <div className="p-6 rounded-xl border border-dark-700 bg-dark-900/50">
        <h3 className="text-lg font-semibold text-dark-50 mb-4">Privacy Budget Tracking</h3>
        <div className="space-y-3">
          <p className="text-dark-400 text-center py-8">No privacy records yet. Run a DP-FedAvg experiment to see privacy metrics.</p>
        </div>
      </div>
    </div>
  )
}
"use client";

import { useStore } from "@/store/useStore";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Trash2, TrendingUp, AlertCircle, Clock } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function HistoryPage() {
  const { history, deleteHistoryEntry, clearHistory } = useStore();

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8 pb-16">
      <div className="flex items-center justify-between mt-2 border-b border-border pb-4">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3 text-text">
            <Clock className="text-accent" /> Assessment History
          </h1>
          <p className="text-muted text-sm mt-1">Review your past credit analyses and scenarios over time.</p>
        </div>
        {history.length > 0 && (
          <Button variant="outline" size="sm" onClick={clearHistory}>
            Clear History
          </Button>
        )}
      </div>

      {history.length === 0 ? (
        <Card className="p-12 text-center flex flex-col items-center justify-center border-dashed border-border">
          <AlertCircle className="text-muted mb-4" size={48} />
          <h3 className="text-xl font-bold mb-2 text-text">No history to display</h3>
          <p className="text-muted mb-6 max-w-sm">
            Run an analysis from the dashboard to start tracking your credit simulation history.
          </p>
          <Button variant="accent" onClick={() => window.location.href = '/home'}>
            Go to Dashboard
          </Button>
        </Card>
      ) : (
        <div className="space-y-4">
          <AnimatePresence>
            {history.map((entry, idx) => (
              <motion.div
                key={entry.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ delay: idx * 0.05 }}
              >
                <Card className="p-5 flex flex-col md:flex-row gap-6 justify-between items-start md:items-center hover:border-accent/40 transition-colors group">
                  <div className="flex-1 space-y-2 w-full text-text">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted font-medium tracking-wide">
                        {new Date(entry.timestamp).toLocaleString(undefined, {
                          dateStyle: "medium",
                          timeStyle: "short",
                        })}
                      </span>
                      {entry.personaName ? (
                        <span className="text-xs font-bold text-accent bg-accent/10 px-2 py-0.5 rounded-full ring-1 ring-accent/20">
                          {entry.personaName}
                        </span>
                      ) : (
                        <span className="text-xs font-bold text-muted bg-border px-2 py-0.5 rounded-full">
                          Manual Entry
                        </span>
                      )}
                    </div>
                    
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-2">
                      <div>
                        <span className="block text-xs text-muted mb-0.5">Income</span>
                        <span className="font-semibold text-sm">${entry.profile.income.toLocaleString()}</span>
                      </div>
                      <div>
                        <span className="block text-xs text-muted mb-0.5">DTI</span>
                        <span className="font-semibold text-sm">{entry.profile.dti}%</span>
                      </div>
                      <div>
                        <span className="block text-xs text-muted mb-0.5">Risk Level</span>
                        <span className={`font-semibold text-sm ${
                          entry.result.risk === 'Low' ? 'text-green-500' :
                          entry.result.risk === 'Medium' ? 'text-yellow-500' : 'text-red-500'
                        }`}>{entry.result.risk}</span>
                      </div>
                      <div>
                        <span className="block text-xs text-muted mb-0.5">Assets</span>
                        <span className="font-semibold text-sm text-text">
                          {entry.profile.ownsHouse ? '🏠' : ''} {entry.profile.ownsCar ? '🚗' : ''}
                          {!entry.profile.ownsHouse && !entry.profile.ownsCar && 'None'}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between md:justify-end gap-6 w-full md:w-auto pt-4 md:pt-0 border-t md:border-t-0 border-border">
                    <div className="text-center">
                      <span className="block text-xs text-muted mb-1">Score Result</span>
                      <div className="text-3xl font-black text-text flex items-center justify-center gap-1">
                        {entry.result.score}
                        <TrendingUp size={16} className={`${entry.result.score > 650 ? 'text-green-500' : 'text-yellow-500'}`} />
                      </div>
                    </div>
                    
                    <button 
                      onClick={() => deleteHistoryEntry(entry.id)}
                      className="h-10 w-10 flex items-center justify-center rounded-xl bg-card border border-border text-muted hover:text-red-500 hover:bg-red-500/10 hover:border-red-500/20 transition-all"
                      title="Delete Entry"
                    >
                      <Trash2 size={18} />
                    </button>
                  </div>
                </Card>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}

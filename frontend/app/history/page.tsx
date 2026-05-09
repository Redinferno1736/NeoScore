"use client";

import { useEffect, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { AlertCircle, Clock, TrendingUp } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";

interface HistoryEntry {
  _id: string;
  score: number;
  risk_tier: string;
  pd: number;
  created_at: string;
}

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/score/history`, { credentials: "include" })
      .then((r) => r.json())
      .then((d) => setHistory(d.events || []))
      .catch(() => setError("Could not load history. Please sign in."))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8 pb-16">
      <div className="flex items-center justify-between mt-2 border-b border-border pb-4">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3 text-text font-serif">
            <Clock className="text-accent" /> Assessment History
          </h1>
          <p className="text-muted text-sm mt-1">
            Review your past credit analyses over time.
          </p>
        </div>
      </div>

      {loading && (
        <div className="flex items-center justify-center py-20">
          <div className="text-muted text-sm animate-pulse">Loading history...</div>
        </div>
      )}

      {error && (
        <Card className="p-12 text-center flex flex-col items-center justify-center border-dashed border-border">
          <AlertCircle className="text-muted mb-4" size={48} />
          <h3 className="text-xl font-bold mb-2 text-text">Unable to load history</h3>
          <p className="text-muted mb-6 max-w-sm">{error}</p>
          <Button variant="accent" onClick={() => window.location.href = "/home"}>
            Go to Dashboard
          </Button>
        </Card>
      )}

      {!loading && !error && history.length === 0 && (
        <Card className="p-12 text-center flex flex-col items-center justify-center border-dashed border-border">
          <AlertCircle className="text-muted mb-4" size={48} />
          <h3 className="text-xl font-bold mb-2 text-text">No history to display</h3>
          <p className="text-muted mb-6 max-w-sm">
            Run an analysis from the dashboard to start tracking your credit simulation history.
          </p>
          <Button variant="accent" onClick={() => window.location.href = "/home"}>
            Go to Dashboard
          </Button>
        </Card>
      )}

      {!loading && !error && history.length > 0 && (
        <div className="space-y-4">
          <AnimatePresence>
            {history.map((entry, idx) => {
              const isLowRisk = ["Excellent", "Very Good", "Good"].includes(entry.risk_tier);
              const riskColor = isLowRisk
                ? "text-green-500"
                : entry.risk_tier === "Fair"
                ? "text-yellow-500"
                : "text-red-500";

              return (
                <motion.div
                  key={entry._id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ delay: idx * 0.05 }}
                >
                  <Card className="p-5 flex flex-col md:flex-row gap-6 justify-between items-start md:items-center hover:border-accent/40 transition-colors group">
                    <div className="flex-1 space-y-2 w-full text-text">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-muted font-medium tracking-wide">
                          {new Date(entry.created_at).toLocaleString(undefined, {
                            dateStyle: "medium",
                            timeStyle: "short",
                          })}
                        </span>
                        <span className={`text-xs font-bold px-2 py-0.5 rounded-full ring-1 ${
                          isLowRisk
                            ? "bg-green-500/10 text-green-500 ring-green-500/20"
                            : entry.risk_tier === "Fair"
                            ? "bg-yellow-500/10 text-yellow-500 ring-yellow-500/20"
                            : "bg-red-500/10 text-red-500 ring-red-500/20"
                        }`}>
                          {entry.risk_tier}
                        </span>
                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 pt-2">
                        <div>
                          <span className="block text-xs text-muted mb-0.5">Risk Tier</span>
                          <span className={`font-semibold text-sm ${riskColor}`}>
                            {entry.risk_tier}
                          </span>
                        </div>
                        <div>
                          <span className="block text-xs text-muted mb-0.5">Default Probability</span>
                          <span className="font-semibold text-sm text-text">
                            {(entry.pd * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center justify-between md:justify-end gap-6 w-full md:w-auto pt-4 md:pt-0 border-t md:border-t-0 border-border">
                      <div className="text-center">
                        <span className="block text-xs text-muted mb-1">Score</span>
                        <div className="text-3xl font-black text-text flex items-center justify-center gap-1">
                          {entry.score}
                          <TrendingUp
                            size={16}
                            className={entry.score > 650 ? "text-green-500" : "text-yellow-500"}
                          />
                        </div>
                      </div>
                    </div>
                  </Card>
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}
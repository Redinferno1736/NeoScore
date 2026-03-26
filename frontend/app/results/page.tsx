"use client";

import { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { useStore } from "@/store/useStore";
import { Card } from "@/components/ui/Card";
import { Slider } from "@/components/ui/Slider";
import { Button } from "@/components/ui/Button";
import { motion, AnimatePresence } from "framer-motion";
import { BrainCircuit, ShieldCheck, AlertTriangle, ChevronRight, Activity, TrendingUp } from "lucide-react";

export default function ResultsPage() {
  const router = useRouter();
  const { profile, result, updateProfileField, predict, isLoading } = useStore();
  const [localProfile, setLocalProfile] = useState(profile);
  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  // If there's no result, they shouldn't be here
  useEffect(() => {
    if (!result && !isLoading) {
      router.push("/home");
    }
  }, [result, isLoading, router]);

  // Debounced prediction when what-if sliders change
  const handleWhatIfChange = (field: keyof typeof profile, value: number) => {
    setLocalProfile((prev) => ({ ...prev, [field]: value }));
    updateProfileField(field, value as never);

    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      predict();
    }, 500);
  };

  if (!result) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-spin text-accent"><Activity size={48} /></div>
      </div>
    );
  }

  const scorePercentage = ((result.score - 300) / 600) * 100;
  const strokeDasharray = 283; 
  const strokeDashoffset = strokeDasharray - (strokeDasharray * scorePercentage) / 100;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8 pb-16">
      <div className="flex items-center justify-between mt-2">
        <h1 className="text-3xl font-bold text-text">Your Credit Profile</h1>
        <Button variant="outline" size="sm" onClick={() => router.push("/home")}>
          Retake Assessment
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Score & Sliders */}
        <div className="lg:col-span-1 space-y-8">
          
          <Card className="p-8 flex flex-col items-center justify-center relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-accent/10 filter blur-[50px] -z-10 rounded-full" />
            <div className="text-center mb-6 z-10">
              <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold ${
                result.risk === 'Low' ? 'bg-green-500/10 text-green-500' :
                result.risk === 'Medium' ? 'bg-yellow-500/10 text-yellow-500' :
                'bg-red-500/10 text-red-500'
              }`}>
                {result.risk === 'Low' ? <ShieldCheck size={14}/> : <AlertTriangle size={14}/>}
                {result.risk} Risk
              </span>
            </div>

            {/* Circular Gauge */}
            <div className="relative w-48 h-48 flex items-center justify-center mb-4 z-10">
              <svg className="w-full h-full transform -rotate-90 pointer-events-none" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="45" fill="none" stroke="currentColor" className="text-border" strokeWidth="8" />
                <motion.circle
                  initial={{ strokeDashoffset: strokeDasharray }}
                  animate={{ strokeDashoffset: isLoading ? strokeDasharray : strokeDashoffset }}
                  transition={{ duration: 1, ease: "easeOut" }}
                  cx="50" cy="50" r="45" fill="none" stroke="currentColor" 
                  className={isLoading ? "text-muted" : "text-accent"} 
                  strokeWidth="8" 
                  strokeLinecap="round"
                  strokeDasharray={strokeDasharray}
                />
              </svg>
              <div className="absolute flex flex-col items-center justify-center">
                <motion.span 
                  className={`text-5xl font-black ${isLoading ? "text-muted" : "text-text"}`}
                  key={result.score}
                  initial={{ scale: 0.9, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                >
                  {result.score}
                </motion.span>
                <span className="text-sm font-medium text-muted">Out of 900</span>
              </div>
            </div>
            
            <div className="w-full pt-4 border-t border-border flex justify-between items-center text-sm z-10">
              <span className="text-muted">Max Possible</span>
              <span className="font-bold text-accent flex items-center gap-1">
                {result.maxScore} <TrendingUp size={14} />
              </span>
            </div>
          </Card>

          {/* Interactive What-If */}
          <Card className="p-6">
            <h3 className="font-bold text-lg mb-6 flex items-center gap-2 text-text">
              <Activity className="text-accent" size={18} /> What-If Simulator
            </h3>
            
            <div className="space-y-6">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted">Income</span>
                  <span className="font-bold text-text">${localProfile.income.toLocaleString()}</span>
                </div>
                <Slider min={15000} max={300000} step={1000} value={localProfile.income} onChange={(v) => handleWhatIfChange("income", v)} />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted">DTI Ratio</span>
                  <span className="font-bold text-text">{localProfile.dti}%</span>
                </div>
                <Slider min={0} max={100} step={1} value={localProfile.dti} onChange={(v) => handleWhatIfChange("dti", v)} />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted">Savings Ratio</span>
                  <span className="font-bold text-text">{localProfile.savingsRatio}%</span>
                </div>
                <Slider min={0} max={100} step={1} value={localProfile.savingsRatio} onChange={(v) => handleWhatIfChange("savingsRatio", v)} />
              </div>
            </div>
          </Card>

        </div>

        {/* Right Column: AI Coach, Feature Impacts & Recommendations */}
        <div className="lg:col-span-2 space-y-6">
          
          <Card className="p-6 border-accent/20 bg-accent/5">
            <div className="flex items-start gap-4">
              <div className="h-10 w-10 flex-shrink-0 rounded-full bg-accent/10 flex items-center justify-center border border-accent/20">
                <BrainCircuit className="text-accent" size={20} />
              </div>
              <div>
                <h3 className="font-bold text-lg mb-1 text-accent">AI Coach Assessment</h3>
                <p className="text-muted leading-relaxed text-sm">
                  {result.coachMessage}
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-6 h-fit">
            <h3 className="font-bold text-lg mb-6 border-b border-border pb-3 text-text">Score Drivers</h3>
            <div className="space-y-4">
              <AnimatePresence mode="popLayout">
                {result.featureImpacts.map((feature, i) => (
                  <motion.div 
                    key={feature.feature}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.1 }}
                    className="flex flex-col gap-2"
                  >
                    <div className="flex justify-between text-sm">
                      <span className="font-medium text-text">{feature.feature}</span>
                      <span className={`font-bold ${feature.positive ? 'text-green-500' : 'text-red-500'}`}>
                        {feature.positive ? '+' : '-'}{feature.impact} pts
                      </span>
                    </div>
                    <div className="w-full bg-border rounded-full h-2 overflow-hidden flex">
                      <div 
                        className={`h-full ${feature.positive ? 'bg-green-500' : 'bg-red-500'}`} 
                        style={{ width: `${Math.min((feature.impact / 50) * 100, 100)}%` }} 
                      />
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </Card>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {result.recommendations.map((rec, i) => (
              <Card key={i} className="p-5 flex flex-col justify-between hover:border-accent/30 transition-colors group cursor-pointer text-text">
                <div>
                  <div className="flex justify-between items-center mb-3">
                    <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full ${
                      rec.effort === 'High' ? 'bg-red-500/10 text-red-500' :
                      rec.effort === 'Medium' ? 'bg-yellow-500/10 text-yellow-500' :
                      'bg-green-500/10 text-green-500'
                    }`}>
                      {rec.effort} Effort
                    </span>
                    <span className="text-green-500 font-bold text-sm">+{rec.impact}</span>
                  </div>
                  <h4 className="font-semibold text-sm leading-tight transition-colors group-hover:text-accent">{rec.action}</h4>
                </div>
                <div className="mt-4 flex justify-end">
                  <ChevronRight size={16} className="text-muted group-hover:text-accent group-hover:translate-x-1 transition-all" />
                </div>
              </Card>
            ))}
          </div>

        </div>
      </div>
    </div>
  );
}

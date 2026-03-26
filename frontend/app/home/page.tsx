"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useStore, Profile } from "@/store/useStore";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Slider } from "@/components/ui/Slider";
import { Toggle } from "@/components/ui/Toggle";
import { Stepper } from "@/components/ui/Stepper";
import { motion } from "framer-motion";

const PERSONAS: {name: string, emoji: string, description: string, range: string, profile: Profile}[] = [
  {
    name: "Salaried Professional",
    emoji: "💼",
    description: "Stable income, low DTI, consistent saver.",
    range: "750 - 850",
    profile: {
      income: 85000,
      dti: 15,
      savingsRatio: 25,
      employmentDuration: "> 5 years",
      age: 34,
      ownsHouse: true,
      ownsCar: true,
      familySize: 2,
      children: 0,
      educationLevel: "Master",
    },
  },
  {
    name: "Recent Graduate",
    emoji: "🎓",
    description: "Low income, no major assets, high potential.",
    range: "550 - 680",
    profile: {
      income: 45000,
      dti: 5,
      savingsRatio: 10,
      employmentDuration: "< 1 year",
      age: 23,
      ownsHouse: false,
      ownsCar: false,
      familySize: 1,
      children: 0,
      educationLevel: "Bachelor",
    },
  },
  {
    name: "Small Business Owner",
    emoji: "🏪",
    description: "Variable income, higher DTI from loans.",
    range: "650 - 750",
    profile: {
      income: 60000,
      dti: 40,
      savingsRatio: 15,
      employmentDuration: "1-3 years",
      age: 42,
      ownsHouse: true,
      ownsCar: true,
      familySize: 4,
      children: 2,
      educationLevel: "Bachelor",
    },
  },
  {
    name: "Financially Struggling",
    emoji: "📉",
    description: "High DTI, low savings, multiple dependents.",
    range: "400 - 550",
    profile: {
      income: 38000,
      dti: 65,
      savingsRatio: 2,
      employmentDuration: "1-3 years",
      age: 38,
      ownsHouse: false,
      ownsCar: true,
      familySize: 5,
      children: 3,
      educationLevel: "High School",
    },
  },
  {
    name: "High Achiever",
    emoji: "🚀",
    description: "High income, vast assets, very low risk.",
    range: "800 - 900",
    profile: {
      income: 250000,
      dti: 8,
      savingsRatio: 40,
      employmentDuration: "> 5 years",
      age: 45,
      ownsHouse: true,
      ownsCar: true,
      familySize: 3,
      children: 1,
      educationLevel: "PhD",
    },
  },
];

export default function HomePage() {
  const router = useRouter();
  const { profile, updateProfileField, setProfile, predict } = useStore();
  const [loadingPersona, setLoadingPersona] = useState<string | null>(null);

  const handlePersonaClick = async (persona: typeof PERSONAS[0]) => {
    setLoadingPersona(persona.name);
    setProfile(persona.profile, persona.name);
    await predict();
    router.push("/results");
  };

  const handleManualSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoadingPersona("manual");
    await predict();
    router.push("/results");
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-12 pb-16">
      <section className="text-center mt-6">
        <h1 className="text-3xl font-bold text-text">
          Choose a starting point
        </h1>
        <p className="text-muted mt-2">
          Select a persona for instant analysis, or enter your own details below.
        </p>
      </section>

      <section>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {PERSONAS.map((persona, idx) => (
            <motion.div
              key={persona.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
            >
              <Card
                className={`p-5 cursor-pointer border-border hover:border-accent/40 transition-all hover:-translate-y-1 ${
                  loadingPersona === persona.name ? "ring-2 ring-accent opacity-80 pointer-events-none" : ""
                }`}
                onClick={() => handlePersonaClick(persona)}
              >
                <div className="flex justify-between items-start mb-2">
                  <span className="text-3xl">{persona.emoji}</span>
                  <span className="text-xs font-bold text-accent bg-accent/10 px-2 py-1 rounded-md">
                    {persona.range}
                  </span>
                </div>
                <h3 className="font-semibold text-lg text-text">{persona.name}</h3>
                <p className="text-xs text-muted mt-1 h-10">{persona.description}</p>
                {loadingPersona === persona.name && (
                  <div className="mt-3 text-xs text-center text-accent animate-pulse font-medium">
                    Analyzing model...
                  </div>
                )}
              </Card>
            </motion.div>
          ))}
        </div>
      </section>

      <div className="flex items-center text-muted">
        <div className="flex-1 border-t border-border"></div>
        <span className="px-4 text-sm font-medium">Or enter your own details</span>
        <div className="flex-1 border-t border-border"></div>
      </div>

      <Card className="p-8">
        <form onSubmit={handleManualSubmit} className="space-y-8">
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
            <div className="space-y-6">
              <h3 className="text-xl font-semibold border-b border-border pb-2 text-accent">Income & Debt</h3>
              
              <div className="space-y-3">
                <div className="flex justify-between items-end">
                  <label className="text-sm font-medium text-muted">Annual Income</label>
                  <span className="text-sm font-bold text-text">${profile.income.toLocaleString()}</span>
                </div>
                <Slider min={15000} max={300000} step={1000} value={profile.income} onChange={(v) => updateProfileField("income", v)} />
              </div>

              <div className="space-y-3">
                <div className="flex justify-between items-end">
                  <label className="text-sm font-medium text-muted">Debt-to-Income (DTI)</label>
                  <span className="text-sm font-bold text-text">{profile.dti}%</span>
                </div>
                <Slider min={0} max={100} step={1} value={profile.dti} onChange={(v) => updateProfileField("dti", v)} />
              </div>

              <div className="space-y-3">
                <div className="flex justify-between items-end">
                  <label className="text-sm font-medium text-muted">Savings Ratio</label>
                  <span className="text-sm font-bold text-text">{profile.savingsRatio}%</span>
                </div>
                <Slider min={0} max={100} step={1} value={profile.savingsRatio} onChange={(v) => updateProfileField("savingsRatio", v)} />
              </div>
            </div>

            <div className="space-y-8">
              <div className="space-y-4">
                <h3 className="text-lg font-semibold border-b border-border pb-2 text-accent">Employment & Age</h3>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-muted">Age</label>
                    <input 
                      type="number" 
                      value={profile.age} 
                      onChange={(e) => updateProfileField("age", Number(e.target.value))} 
                      className="w-full bg-bg border border-border rounded-xl px-3 py-2 text-text focus:outline-none focus:border-accent transition-colors"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-muted">Job Duration</label>
                    <select 
                      value={profile.employmentDuration} 
                      onChange={(e) => updateProfileField("employmentDuration", e.target.value)}
                      className="w-full bg-bg border border-border rounded-xl px-3 py-2 text-text focus:outline-none focus:border-accent appearance-none transition-colors"
                    >
                      <option>&lt; 1 year</option>
                      <option>1-3 years</option>
                      <option>3-5 years</option>
                      <option>&gt; 5 years</option>
                    </select>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                 <h3 className="text-lg font-semibold border-b border-border pb-2 text-accent">Assets</h3>
                 <div className="flex gap-8">
                   <div className="flex items-center space-x-3">
                     <Toggle checked={profile.ownsHouse} onChange={(v) => updateProfileField("ownsHouse", v)} />
                     <span className="text-sm font-medium text-text">Owns House</span>
                   </div>
                   <div className="flex items-center space-x-3">
                     <Toggle checked={profile.ownsCar} onChange={(v) => updateProfileField("ownsCar", v)} />
                     <span className="text-sm font-medium text-text">Owns Car</span>
                   </div>
                 </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-lg font-semibold border-b border-border pb-2 text-accent">Personal</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-muted">Family Size</label>
                    <Stepper value={profile.familySize} min={1} max={10} onChange={(v) => updateProfileField("familySize", v)} />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-muted">Children</label>
                    <Stepper value={profile.children} min={0} max={10} onChange={(v) => updateProfileField("children", v)} />
                  </div>
                </div>
                <div className="space-y-2 pt-2">
                  <label className="text-sm font-medium text-muted block">Education Level</label>
                  <select 
                    value={profile.educationLevel} 
                    onChange={(e) => updateProfileField("educationLevel", e.target.value)}
                    className="w-full bg-bg border border-border rounded-xl px-3 py-2 text-text focus:outline-none focus:border-accent appearance-none transition-colors"
                  >
                    <option>High School</option>
                    <option>Bachelor</option>
                    <option>Master</option>
                    <option>PhD</option>
                  </select>
                </div>
              </div>

            </div>
          </div>
          
          <div className="pt-6 border-t border-border text-right">
            <Button 
              type="submit" 
              variant="accent" 
              size="lg" 
              className="w-full sm:w-64"
              disabled={loadingPersona !== null}
            >
              {loadingPersona === "manual" ? "Generating Score..." : "Calculate Score"}
            </Button>
          </div>
        </form>
      </Card>
    </div>
  );
}

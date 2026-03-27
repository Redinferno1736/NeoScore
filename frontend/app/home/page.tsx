"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useStore, RawProfile } from "@/store/useStore";
import { motion } from "framer-motion";

// ─── Demo Personas ─────────────────────────────────────────────────────────────
const DEMO_PERSONAS: {
  name: string; emoji: string; description: string; range: string; profile: RawProfile;
}[] = [
  {
    name: "Ravi", emoji: "🛵",
    description: "Gig worker, thin credit file, young.",
    range: "480 – 560",
    profile: {
      AMT_INCOME_TOTAL: 120000, AMT_CREDIT: 100000, AMT_ANNUITY: 7000, AMT_GOODS_PRICE: 90000,
      AGE_YEARS: 28, CNT_CHILDREN: 1, CNT_FAM_MEMBERS: 3, EMPLOYED_YEARS: 0.8,
      NAME_INCOME_TYPE: "Working", OCCUPATION_TYPE: "Laborers",
      ORGANIZATION_TYPE: "Business Entity Type 3",
      NAME_EDUCATION_TYPE: "Secondary / secondary special", NAME_FAMILY_STATUS: "Married",
      FLAG_OWN_REALTY: 0, FLAG_OWN_CAR: 0, NAME_HOUSING_TYPE: "House / apartment",
      REGION_RATING_CLIENT: 2, REGION_POPULATION_RELATIVE: 0.018,
    },
  },
  {
    name: "Priya", emoji: "💼",
    description: "Salaried professional, stable income.",
    range: "680 – 760",
    profile: {
      AMT_INCOME_TOTAL: 300000, AMT_CREDIT: 250000, AMT_ANNUITY: 15000, AMT_GOODS_PRICE: 230000,
      AGE_YEARS: 32, CNT_CHILDREN: 0, CNT_FAM_MEMBERS: 2, EMPLOYED_YEARS: 5,
      NAME_INCOME_TYPE: "Commercial associate", OCCUPATION_TYPE: "Managers",
      ORGANIZATION_TYPE: "Business Entity Type 2",
      NAME_EDUCATION_TYPE: "Higher education", NAME_FAMILY_STATUS: "Single / not married",
      FLAG_OWN_REALTY: 1, FLAG_OWN_CAR: 0, NAME_HOUSING_TYPE: "House / apartment",
      REGION_RATING_CLIENT: 2, REGION_POPULATION_RELATIVE: 0.035,
    },
  },
  {
    name: "Deepa", emoji: "🏪",
    description: "Self-employed, moderate risk.",
    range: "580 – 650",
    profile: {
      AMT_INCOME_TOTAL: 200000, AMT_CREDIT: 400000, AMT_ANNUITY: 16000, AMT_GOODS_PRICE: 380000,
      AGE_YEARS: 38, CNT_CHILDREN: 2, CNT_FAM_MEMBERS: 4, EMPLOYED_YEARS: 2.5,
      NAME_INCOME_TYPE: "Working", OCCUPATION_TYPE: "Laborers",
      ORGANIZATION_TYPE: "Self-employed",
      NAME_EDUCATION_TYPE: "Higher education", NAME_FAMILY_STATUS: "Married",
      FLAG_OWN_REALTY: 1, FLAG_OWN_CAR: 0, NAME_HOUSING_TYPE: "House / apartment",
      REGION_RATING_CLIENT: 2, REGION_POPULATION_RELATIVE: 0.025,
    },
  },
  {
    name: "High Achiever", emoji: "🚀",
    description: "High income, strong assets, very low risk.",
    range: "800 – 900",
    profile: {
      AMT_INCOME_TOTAL: 600000, AMT_CREDIT: 300000, AMT_ANNUITY: 18000, AMT_GOODS_PRICE: 280000,
      AGE_YEARS: 45, CNT_CHILDREN: 1, CNT_FAM_MEMBERS: 3, EMPLOYED_YEARS: 15,
      NAME_INCOME_TYPE: "State servant", OCCUPATION_TYPE: "Managers",
      ORGANIZATION_TYPE: "Government",
      NAME_EDUCATION_TYPE: "Higher education", NAME_FAMILY_STATUS: "Married",
      FLAG_OWN_REALTY: 1, FLAG_OWN_CAR: 1, NAME_HOUSING_TYPE: "House / apartment",
      REGION_RATING_CLIENT: 1, REGION_POPULATION_RELATIVE: 0.04,
    },
  },
];

// ─── Dropdown options (exact Home Credit class names) ─────────────────────────
const EDUCATION_OPTIONS = [
  "Lower secondary", "Secondary / secondary special", "Incomplete higher",
  "Higher education", "Academic degree",
];
const FAMILY_OPTIONS = [
  "Single / not married", "Married", "Civil marriage", "Separated", "Widow",
];
const INCOME_TYPE_OPTIONS = [
  "Working", "Commercial associate", "State servant", "Pensioner",
  "Businessman", "Student", "Unemployed", "Maternity leave",
];
const OCCUPATION_OPTIONS = [
  "Laborers", "Core staff", "Accountants", "Managers", "Drivers", "Sales staff",
  "Cleaning staff", "Cooking staff", "Private service staff", "Medicine staff",
  "Security staff", "High skill tech staff", "Waiters/barmen staff",
  "Low-skill Laborers", "Secretaries", "Realty agents", "HR staff", "IT staff",
];
const ORGANIZATION_OPTIONS = [
  "Business Entity Type 3", "Business Entity Type 2", "Business Entity Type 1",
  "School", "Government", "Religion", "Other", "Medicine", "Self-employed",
  "Transport: type 2", "Transport: type 3", "Industry: type 1", "Industry: type 9",
  "Trade: type 7", "Telecom", "Construction", "Military", "Bank",
];
const HOUSING_OPTIONS = [
  "House / apartment", "With parents", "Municipal apartment",
  "Rented apartment", "Office apartment", "Co-op apartment",
];

// ─── Tiny UI primitives ───────────────────────────────────────────────────────
function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-xs font-bold text-accent uppercase tracking-widest pb-2 border-b border-border/60 mb-4">
      {children}
    </h3>
  );
}
function Label({ children }: { children: React.ReactNode }) {
  return (
    <label className="block text-xs font-semibold text-muted uppercase tracking-widest mb-1.5">
      {children}
    </label>
  );
}
function NumInput({ value, onChange, min = 0, max, step = 1, prefix }: {
  value: number; onChange: (v: number) => void;
  min?: number; max?: number; step?: number; prefix?: string;
}) {
  return (
    <div className="flex items-center bg-bg border border-border rounded-xl overflow-hidden focus-within:border-accent transition-colors">
      {prefix && (
        <span className="px-3 py-2.5 text-xs text-muted border-r border-border bg-surface select-none">
          {prefix}
        </span>
      )}
      <input type="number" value={value} min={min} max={max} step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="flex-1 bg-transparent px-3 py-2.5 text-text text-sm focus:outline-none min-w-0" />
    </div>
  );
}
function SelInput({ value, onChange, options }: {
  value: string; onChange: (v: string) => void; options: string[];
}) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}
      className="w-full bg-bg border border-border rounded-xl px-3 py-2.5 text-text text-sm
                 focus:outline-none focus:border-accent transition-colors appearance-none cursor-pointer">
      {options.map((o) => <option key={o} value={o}>{o}</option>)}
    </select>
  );
}
function Toggle({ checked, onChange, label }: {
  checked: boolean; onChange: (v: boolean) => void; label: string;
}) {
  return (
    <button type="button" onClick={() => onChange(!checked)}
      className={`flex items-center gap-3 px-4 py-3 rounded-xl border transition-all text-sm font-medium
        ${checked ? "border-accent bg-accent/10 text-accent" : "border-border bg-bg text-muted hover:border-accent/40"}`}>
      <span className={`w-9 h-5 rounded-full relative transition-colors flex-shrink-0 ${checked ? "bg-accent" : "bg-border"}`}>
        <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform
          ${checked ? "translate-x-4" : "translate-x-0.5"}`} />
      </span>
      {label}
    </button>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────
export default function HomePage() {
  const router = useRouter();

  // ✅ Single source of truth — store owns profile + predict()
  const { profile, setProfile, updateProfileField, predict, isLoading, error, clearError } =
    useStore();

  const [loadingLabel, setLoadingLabel] = useState<string | null>(null);

  const set = <K extends keyof RawProfile>(key: K, val: RawProfile[K]) =>
    updateProfileField(key, val);

  // ─── Unified predict flow ─────────────────────────────────────────────────
  const runPredict = async (p: RawProfile, label: string) => {
    setLoadingLabel(label);
    clearError();
    setProfile(p, label);   // write to store first

    try {
      await predict();      // store.predict() sends store.profile → backend
    } catch {
      // store.error is already set; just stop button spinner
      setLoadingLabel(null);
      return;               // don't navigate on failure
    }

    router.push("/results");
  };

  const handlePersonaClick = (persona: typeof DEMO_PERSONAS[0]) =>
    runPredict(persona.profile, persona.name);

  const handleManualSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    runPredict(profile, "manual");
  };

  const busy = isLoading || loadingLabel !== null;

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-10 pb-20">

      {/* Hero */}
      <section className="text-center pt-4">
        <h1 className="text-3xl font-bold text-text">NeoScore Credit Analyser</h1>
        <p className="text-muted mt-2 text-sm">
          Pick a demo profile for instant analysis, or enter your own details below.
        </p>
      </section>

      {/* Error banner — from store */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 rounded-xl px-4 py-3 text-sm text-center">
          {error}
        </div>
      )}

      {/* Personas */}
      <section>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {DEMO_PERSONAS.map((p, i) => (
            <motion.button key={p.name} type="button"
              initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.07 }}
              onClick={() => handlePersonaClick(p)} disabled={busy}
              className={`text-left p-4 rounded-xl border transition-all
                ${loadingLabel === p.name
                  ? "border-accent bg-accent/10 opacity-80 pointer-events-none"
                  : "border-border hover:border-accent/50 hover:-translate-y-0.5 bg-surface"
                } ${busy && loadingLabel !== p.name ? "opacity-50 cursor-not-allowed" : ""}`}>
              <div className="flex justify-between items-start mb-2">
                <span className="text-2xl">{p.emoji}</span>
                <span className="text-[10px] font-bold text-accent bg-accent/10 px-1.5 py-0.5 rounded">
                  {p.range}
                </span>
              </div>
              <div className="font-semibold text-sm text-text">{p.name}</div>
              <div className="text-xs text-muted mt-0.5 leading-snug">{p.description}</div>
              {loadingLabel === p.name && (
                <div className="text-[10px] text-accent animate-pulse mt-2 font-medium">Analysing…</div>
              )}
            </motion.button>
          ))}
        </div>
      </section>

      <div className="flex items-center text-muted">
        <div className="flex-1 border-t border-border" />
        <span className="px-4 text-xs font-semibold uppercase tracking-widest">Or enter your details</span>
        <div className="flex-1 border-t border-border" />
      </div>

      {/* ── Manual form ──────────────────────────────────────────────────────── */}
      <form onSubmit={handleManualSubmit}>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* LEFT */}
          <div className="space-y-6">
            {/* Loan Details */}
            <div className="bg-surface border border-border rounded-2xl p-6">
              <SectionTitle>Loan Details</SectionTitle>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Annual Income (₹)</Label>
                  <NumInput prefix="₹" value={profile.AMT_INCOME_TOTAL} min={10000} max={10000000} step={5000}
                    onChange={(v) => set("AMT_INCOME_TOTAL", v)} />
                </div>
                <div>
                  <Label>Loan Amount (₹)</Label>
                  <NumInput prefix="₹" value={profile.AMT_CREDIT} min={10000} max={10000000} step={5000}
                    onChange={(v) => set("AMT_CREDIT", v)} />
                </div>
                <div>
                  <Label>Monthly EMI (₹)</Label>
                  <NumInput prefix="₹" value={profile.AMT_ANNUITY} min={500} max={500000} step={500}
                    onChange={(v) => set("AMT_ANNUITY", v)} />
                </div>
                <div>
                  <Label>Goods / Asset Price (₹)</Label>
                  <NumInput prefix="₹" value={profile.AMT_GOODS_PRICE} min={10000} max={10000000} step={5000}
                    onChange={(v) => set("AMT_GOODS_PRICE", v)} />
                </div>
              </div>
              <div className="mt-4 text-xs text-muted bg-bg rounded-lg px-3 py-2 flex flex-wrap gap-4">
                <span>Loan/Income: <b className="text-text">{(profile.AMT_CREDIT / (profile.AMT_INCOME_TOTAL || 1)).toFixed(2)}×</b></span>
                <span>EMI/Income: <b className="text-text">{((profile.AMT_ANNUITY / (profile.AMT_INCOME_TOTAL || 1)) * 100).toFixed(1)}%</b></span>
                <span>Coverage: <b className="text-text">{(profile.AMT_INCOME_TOTAL / (profile.AMT_ANNUITY * 12 || 1)).toFixed(1)}×</b></span>
              </div>
            </div>

            {/* Employment */}
            <div className="bg-surface border border-border rounded-2xl p-6">
              <SectionTitle>Employment</SectionTitle>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Years Employed</Label>
                  <NumInput value={profile.EMPLOYED_YEARS} min={0} max={50} step={0.5}
                    onChange={(v) => set("EMPLOYED_YEARS", v)} />
                </div>
                <div>
                  <Label>Income Type</Label>
                  <SelInput value={profile.NAME_INCOME_TYPE} options={INCOME_TYPE_OPTIONS}
                    onChange={(v) => set("NAME_INCOME_TYPE", v)} />
                </div>
                <div>
                  <Label>Occupation</Label>
                  <SelInput value={profile.OCCUPATION_TYPE} options={OCCUPATION_OPTIONS}
                    onChange={(v) => set("OCCUPATION_TYPE", v)} />
                </div>
                <div>
                  <Label>Organisation Type</Label>
                  <SelInput value={profile.ORGANIZATION_TYPE} options={ORGANIZATION_OPTIONS}
                    onChange={(v) => set("ORGANIZATION_TYPE", v)} />
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT */}
          <div className="space-y-6">
            {/* Personal */}
            <div className="bg-surface border border-border rounded-2xl p-6">
              <SectionTitle>Personal</SectionTitle>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <Label>Age</Label>
                  <NumInput value={profile.AGE_YEARS} min={18} max={80}
                    onChange={(v) => set("AGE_YEARS", v)} />
                </div>
                <div>
                  <Label>Children</Label>
                  <NumInput value={profile.CNT_CHILDREN} min={0} max={15}
                    onChange={(v) => set("CNT_CHILDREN", v)} />
                </div>
                <div>
                  <Label>Family Size</Label>
                  <NumInput value={profile.CNT_FAM_MEMBERS} min={1} max={20}
                    onChange={(v) => set("CNT_FAM_MEMBERS", v)} />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4 mt-4">
                <div>
                  <Label>Education</Label>
                  <SelInput value={profile.NAME_EDUCATION_TYPE} options={EDUCATION_OPTIONS}
                    onChange={(v) => set("NAME_EDUCATION_TYPE", v)} />
                </div>
                <div>
                  <Label>Marital Status</Label>
                  <SelInput value={profile.NAME_FAMILY_STATUS} options={FAMILY_OPTIONS}
                    onChange={(v) => set("NAME_FAMILY_STATUS", v)} />
                </div>
              </div>
            </div>

            {/* Assets & Housing */}
            <div className="bg-surface border border-border rounded-2xl p-6">
              <SectionTitle>Assets &amp; Housing</SectionTitle>
              <div className="flex gap-3 flex-wrap mb-4">
                <Toggle checked={profile.FLAG_OWN_REALTY === 1} label="Owns Property"
                  onChange={(v) => set("FLAG_OWN_REALTY", v ? 1 : 0)} />
                <Toggle checked={profile.FLAG_OWN_CAR === 1} label="Owns Car"
                  onChange={(v) => set("FLAG_OWN_CAR", v ? 1 : 0)} />
              </div>
              <Label>Housing Type</Label>
              <SelInput value={profile.NAME_HOUSING_TYPE} options={HOUSING_OPTIONS}
                onChange={(v) => set("NAME_HOUSING_TYPE", v)} />
            </div>

            {/* Region */}
            <div className="bg-surface border border-border rounded-2xl p-6">
              <SectionTitle>Region</SectionTitle>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Region Credit Rating</Label>
                  <div className="flex gap-2">
                    {[1, 2, 3].map((r) => (
                      <button key={r} type="button"
                        onClick={() => set("REGION_RATING_CLIENT", r)}
                        className={`flex-1 py-2.5 rounded-xl border text-sm font-bold transition-all
                          ${profile.REGION_RATING_CLIENT === r
                            ? "border-accent bg-accent/10 text-accent"
                            : "border-border text-muted hover:border-accent/40"}`}>
                        {r === 1 ? "1 ★" : r === 2 ? "2" : "3 ▼"}
                      </button>
                    ))}
                  </div>
                  <p className="text-[10px] text-muted mt-1">1 = best rated region</p>
                </div>
                <div>
                  <Label>Population Density</Label>
                  <NumInput value={profile.REGION_POPULATION_RELATIVE} min={0.001} max={0.1} step={0.001}
                    onChange={(v) => set("REGION_POPULATION_RELATIVE", v)} />
                  <p className="text-[10px] text-muted mt-1">Relative (0.001 – 0.072)</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Submit */}
        <div className="mt-6 text-right">
          <button type="submit" disabled={busy}
            className={`px-10 py-3.5 rounded-xl font-semibold text-sm tracking-wide transition-all
              ${busy
                ? "bg-accent/40 text-white cursor-not-allowed"
                : "bg-accent text-white hover:bg-accent/90 hover:shadow-lg hover:shadow-accent/20 active:scale-95"}`}>
            {loadingLabel === "manual" ? "Calculating…" : "Calculate My Score →"}
          </button>
        </div>
      </form>
    </div>
  );
}
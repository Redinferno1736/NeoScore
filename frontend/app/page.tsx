"use client";

import { motion, type Transition } from "framer-motion";
import Link from "next/link";
import { Button } from "@/components/ui/Button";
import Image from "next/image";
import { useEffect, useState } from "react";
import { ArrowRight, TrendingUp, ShieldCheck, Sparkles } from "lucide-react";

/* ── Animated credit score ring ─────────────────────────────────────── */
function ScoreRing() {
  const [score, setScore] = useState(0);
  const target = 742;
  const max = 850;
  const r = 68;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - score / max);

  useEffect(() => {
    const t = setTimeout(() => setScore(target), 800);
    return () => clearTimeout(t);
  }, []);

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width="168" height="168" className="-rotate-90" aria-hidden>
        <circle cx="84" cy="84" r={r} fill="none" strokeWidth="7"
          stroke="rgba(227,208,180,0.10)" />
        <circle cx="84" cy="84" r={r} fill="none" strokeWidth="7"
          stroke="var(--accent)" strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 1.6s cubic-bezier(.16,1,.3,1)" }}
        />
      </svg>
      <div className="absolute text-center select-none">
        <div className="text-4xl font-serif font-semibold" style={{ color: "#F8FAFC" }}>
          {score}
        </div>
        <div className="text-[10px] uppercase tracking-[0.18em] mt-0.5" style={{ color: "var(--accent)" }}>
          Score
        </div>
      </div>
    </div>
  );
}

/* ── Factor bar ─────────────────────────────────────────────────────── */
function FactorBar({ label, pct, delay }: { label: string; pct: number; delay: number }) {
  const [w, setW] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setW(pct), 900 + delay);
    return () => clearTimeout(t);
  }, [pct, delay]);

  return (
    <div className="flex items-center gap-3">
      <div className="text-[11px] uppercase tracking-widest w-36 shrink-0" style={{ color: "#A8A8A8" }}>{label}</div>
      <div className="flex-1 h-[3px] rounded-full" style={{ background: "rgba(227,208,180,0.12)" }}>
        <div className="h-full rounded-full"
          style={{
            width: `${w}%`,
            background: "var(--accent)",
            transition: `width 1.2s cubic-bezier(.16,1,.3,1) ${delay}ms`
          }} />
      </div>
      <div className="text-[11px] font-mono w-8 text-right" style={{ color: "var(--accent)" }}>{pct}%</div>
    </div>
  );
}

/* ── Main page ───────────────────────────────────────────────────────── */
export default function LandingPage() {
  const [liveUsers, setLiveUsers] = useState(12438);
  const [activeTab, setActiveTab] = useState("Score");

  const handleLogin = () => {
      window.location.href = "http://localhost:5000/auth/login/google";
  };

  useEffect(() => {
    const iv = setInterval(() => setLiveUsers(p => p + Math.floor(Math.random() * 3)), 4000);
    return () => clearInterval(iv);
  }, []);


  

  const fadeUp = (delay = 0) => {
    const ease = [0.16, 1, 0.3, 1] as [number, number, number, number];
    const base: Transition = { duration: 1.1, ease, delay };
    return {
      initial: { opacity: 0, y: 28 },
      animate: { opacity: 1, y: 0 },
      transition: base,
    };
  };

  const tabs = ["Score", "Analysis", "Forecast"];

  return (
    <div className="relative w-full">
      {/* ── Background image with legibility overlays ────────────── */}
      <div className="fixed inset-0 -z-10 pointer-events-none" aria-hidden="true">
        {/* Photo */}
        <Image
          src="/bg.jpg"
          alt=""
          fill
          priority
          className="object-cover object-center"
          quality={90}
        />
        {/* Layer 1: base dark tint — kills ambient brightness */}
        <div className="absolute inset-0" style={{ background: "rgba(4,12,10,0.72)" }} />
        {/* Layer 2: gradient — darkens top & bottom harder, mid stays lighter */}
        <div className="absolute inset-0" style={{
          background: "linear-gradient(180deg, rgba(4,12,10,0.55) 0%, rgba(4,12,10,0.15) 40%, rgba(4,12,10,0.55) 80%, rgba(4,12,10,0.90) 100%)"
        }} />
        {/* Layer 3: brand color bleed — ties the photo into the Obsidian palette */}
        <div className="absolute inset-0" style={{ background: "rgba(8,20,18,0.38)" }} />
      </div>


      {/* ── HERO ─────────────────────────────────────────────── */}
      <section className="relative z-10 w-full min-h-[calc(100vh-4rem)] grid grid-cols-1 lg:grid-cols-[1fr_1.15fr] gap-0 items-center px-6 lg:px-0">

        {/* LEFT — editorial headline */}
        <div className="lg:pl-16 xl:pl-24 flex flex-col justify-center gap-8 py-20 lg:py-0">
          {/* Label */}
          <motion.div {...fadeUp(0.1)} className="flex items-center gap-3">
            <span className="text-accent font-mono text-xs tracking-[0.22em] uppercase opacity-80">
              / AI Credit Intelligence /
            </span>
          </motion.div>

          {/* Giant headline */}
          <div className="overflow-hidden">
            <motion.h1
              initial={{ opacity: 0, y: 60 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1], delay: 0.15 }}
              className="font-serif leading-[0.92] tracking-tighter"
              style={{
                fontSize: "clamp(3.8rem, 8vw, 7rem)",
                color: "#F8FAFC",
              }}
            >
              THE<br />
              NEW.<br />
              <span style={{ color: "var(--accent)" }}>LEVEL&nbsp;OF.</span><br />
              CREDIT.
            </motion.h1>
          </div>

          {/* Tagline */}
          <motion.p {...fadeUp(0.4)}
            className="max-w-sm text-sm leading-relaxed"
            style={{ color: "#A8A8A8", fontFamily: "var(--font-sans)" }}
          >
            Explainable AI credit scoring. Know exactly what drives your number — and precisely how to move it.
          </motion.p>

          {/* CTA row */}
          <motion.div {...fadeUp(0.55)} className="flex items-center gap-5">
            {/* Replaced <Link> with onClick handler */}
            <Button onClick={handleLogin} variant="accent" size="lg" className="group">
              SIGN IN WITH GOOGLE
              <ArrowRight size={14} className="ml-2 group-hover:translate-x-1 transition-transform duration-200" />
            </Button>
            
            <button className="text-xs uppercase tracking-[0.18em] font-semibold flex items-center gap-2 transition-opacity hover:opacity-60"
              style={{ color: "#A8A8A8" }}>
              View Demo
            </button>
          </motion.div>

          {/* Social proof */}
          <motion.div {...fadeUp(0.7)} className="flex items-center gap-4 pt-2">
            {/* Avatar stack */}
            <div className="flex -space-x-2">
              {["#5C8A62", "#8A7E5C", "#5C6A8A"].map((c, i) => (
                <div key={i} className="w-7 h-7 rounded-full border-2 border-[rgba(8,20,18,0.8)] flex items-center justify-center text-[10px] font-bold"
                  style={{ background: c, color: "#F8FAFC" }}>
                  {["A", "R", "K"][i]}
                </div>
              ))}
            </div>
            <div>
              <div className="text-sm font-semibold" style={{ color: "#F8FAFC" }}>
                {liveUsers.toLocaleString()}+
              </div>
              <div className="text-[10px] uppercase tracking-widest" style={{ color: "#A8A8A8" }}>
                Active Users
              </div>
            </div>
          </motion.div>
        </div>

        {/* RIGHT — floating product card */}
        <motion.div
          initial={{ opacity: 0, x: 40 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 1.3, ease: [0.16, 1, 0.3, 1], delay: 0.3 }}
          className="flex items-center justify-center py-16 lg:py-8 lg:pr-12 xl:pr-20"
        >
          <div
            className="w-full max-w-md relative"
            style={{
              background: "linear-gradient(145deg, rgba(8,20,18,0.96) 0%, rgba(14,32,28,0.92) 100%)",
              border: "1px solid rgba(227,208,180,0.22)",
              backdropFilter: "blur(24px)",
              WebkitBackdropFilter: "blur(24px)",
              borderRadius: "2px",
            }}
          >
            {/* Card header — tab row */}
            <div className="flex items-center gap-2 px-7 pt-7 pb-5 border-b"
              style={{ borderColor: "rgba(227,208,180,0.10)" }}>
              {tabs.map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className="text-[11px] uppercase tracking-[0.14em] px-4 py-1.5 transition-all duration-200"
                  style={{
                    border: "1px solid",
                    borderRadius: "999px",
                    borderColor: activeTab === tab ? "var(--accent)" : "rgba(227,208,180,0.20)",
                    color: activeTab === tab ? "var(--accent)" : "#A8A8A8",
                    background: activeTab === tab ? "rgba(227,208,180,0.06)" : "transparent",
                  }}
                >
                  {tab}
                </button>
              ))}

              {/* Corner label — like "ROOMTOUR" */}
              <div className="ml-auto text-[10px] uppercase tracking-[0.18em] px-3 py-1"
                style={{
                  background: "rgba(227,208,180,0.10)",
                  border: "1px solid rgba(227,208,180,0.20)",
                  color: "var(--accent)",
                  borderRadius: "2px",
                }}>
                LIVE DATA
              </div>
            </div>

            {/* Card body */}
            <div className="px-7 py-7 flex flex-col gap-7">
              {/* Score ring + label */}
              <div className="flex items-center gap-8">
                <ScoreRing />
                <div className="flex flex-col gap-3">
                  <div>
                    <div className="text-[10px] uppercase tracking-[0.18em] mb-1" style={{ color: "#A8A8A8" }}>
                      Credit Rating
                    </div>
                    <div className="text-xl font-serif font-semibold" style={{ color: "#F8FAFC" }}>
                      Excellent
                    </div>
                  </div>
                  <div
                    className="text-[11px] leading-relaxed"
                    style={{ color: "#A8A8A8", maxWidth: "160px" }}
                  >
                    Top 11% of all NeoScore users nationally.
                  </div>
                  <div className="inline-flex items-center gap-1.5 text-[11px]"
                    style={{ color: "#6ECA8F" }}>
                    <TrendingUp size={12} />
                    <span>+18 pts this month</span>
                  </div>
                </div>
              </div>

              {/* Factor bars */}
              <div className="flex flex-col gap-3.5">
                <FactorBar label="Payment History" pct={98} delay={0} />
                <FactorBar label="Credit Utilization" pct={23} delay={80} />
                <FactorBar label="Credit Age" pct={71} delay={160} />
                <FactorBar label="Account Mix" pct={85} delay={240} />
              </div>

              {/* Card footer */}
              <div className="flex items-center justify-between pt-1 border-t"
                style={{ borderColor: "rgba(227,208,180,0.08)" }}>
                <div className="flex items-center gap-2 text-[11px]" style={{ color: "#A8A8A8" }}>
                  <ShieldCheck size={12} style={{ color: "var(--accent)" }} />
                  Bank-level encryption
                </div>
                <Link href="/home"
                  className="flex items-center gap-1.5 text-[11px] uppercase tracking-widest font-semibold transition-opacity hover:opacity-70"
                  style={{ color: "var(--accent)" }}>
                  Full Dashboard
                  <ArrowRight size={11} />
                </Link>
              </div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* ── BOTTOM STRIP ─────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] as [number, number, number, number], delay: 0.9 }}
        className="relative z-10 w-full grid grid-cols-1 md:grid-cols-3 border-t"
        style={{ borderColor: "rgba(227,208,180,0.12)" }}
      >
        {/* Stat 1 */}
        <div
          className="px-8 py-8 flex flex-col gap-2 border-b md:border-b-0 md:border-r"
          style={{
            borderColor: "rgba(227,208,180,0.12)",
            background: "linear-gradient(135deg, rgba(8,20,18,0.92) 0%, rgba(8,20,18,0.80) 100%)",
          }}
        >
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.18em]" style={{ color: "var(--accent)" }}>
            <Sparkles size={11} />
            AI-Powered
          </div>
          <div className="text-3xl font-serif" style={{ color: "#F8FAFC" }}>
            98.4%
          </div>
          <p className="text-xs leading-relaxed" style={{ color: "#A8A8A8" }}>
            Model accuracy on score prediction across all user cohorts.
          </p>
        </div>

        {/* Stat 2 — center */}
        <div
          className="px-8 py-8 flex flex-col justify-between gap-6 border-b md:border-b-0 md:border-r"
          style={{
            borderColor: "rgba(227,208,180,0.12)",
            background: "rgba(8,20,18,0.70)",
          }}
        >
          <div className="text-[10px] uppercase tracking-[0.18em]" style={{ color: "#A8A8A8" }}>
            Community
          </div>
          <div>
            <div className="text-4xl font-serif italic" style={{ color: "var(--accent)" }}>
              12k+
            </div>
            <div className="text-xs uppercase tracking-widest mt-1" style={{ color: "#A8A8A8" }}>
              Customers
            </div>
          </div>
          <p className="text-xs leading-relaxed" style={{ color: "#A8A8A8" }}>
            Improving their financial health with NeoScore&apos;s explainable AI.
          </p>
        </div>

        {/* Stat 3 — headline block */}
        <div
          className="px-8 py-8 flex flex-col justify-between gap-6"
          style={{ background: "rgba(8,20,18,0.60)" }}
        >
          <div className="text-[10px] uppercase tracking-[0.18em]" style={{ color: "var(--accent)" }}>
            Our Promise
          </div>
          <h2 className="font-serif leading-tight" style={{ fontSize: "clamp(1.2rem, 2vw, 1.6rem)", color: "#F8FAFC" }}>
            WE MAKE YOUR CREDIT SCORE WORK FOR YOU.
          </h2>
          <Link href="/home"
            className="flex items-center gap-2 text-[11px] uppercase tracking-widest font-semibold transition-opacity hover:opacity-60"
            style={{ color: "var(--accent)" }}>
            Learn More <ArrowRight size={12} />
          </Link>
        </div>
      </motion.div>
    </div>
  );
}

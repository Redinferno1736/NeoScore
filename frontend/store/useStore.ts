/**
 * useStore.ts — NeoScore global state
 *
 * Schema is aligned 1:1 with backend STUDENT_FEATURES.
 * No translation layer needed — profile fields ARE the API payload.
 *
 * FIXES:
 *  1. predict() accepts optional profileOverride so home/page.tsx can
 *     pass the profile directly — avoids race condition where get().profile
 *     hasn't updated yet when predict() runs after setProfile()
 *  2. Removed credentials:"include" from /api/score (public endpoint)
 *  3. On error, result is set to null so results page redirects correctly
 *  4. Better error messages for debugging
 */

import { create } from "zustand";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface RawProfile {
  // Loan
  AMT_INCOME_TOTAL: number;
  AMT_CREDIT: number;
  AMT_ANNUITY: number;
  AMT_GOODS_PRICE: number;
  // Personal
  AGE_YEARS: number;
  CNT_CHILDREN: number;
  CNT_FAM_MEMBERS: number;
  // Employment
  EMPLOYED_YEARS: number;
  NAME_INCOME_TYPE: string;
  OCCUPATION_TYPE: string;
  ORGANIZATION_TYPE: string;
  // Education & family
  NAME_EDUCATION_TYPE: string;
  NAME_FAMILY_STATUS: string;
  // Assets
  FLAG_OWN_REALTY: 0 | 1;
  FLAG_OWN_CAR: 0 | 1;
  NAME_HOUSING_TYPE: string;
  // Region
  REGION_RATING_CLIENT: number;
  REGION_POPULATION_RELATIVE: number;
}

// Keep legacy alias so any existing import of `Profile` keeps compiling
export type Profile = RawProfile;

interface ScoreResult {
  score: number;
  risk_tier: string;
  approval_probability: number;

  ai_explanation?: string;
  reasoning?: string;
  decision_reasoning?: string;

  rule_triggered?: string;
  confidence_note?: string;

  explanation?: {
    narrative?: string;
    drivers_positive?: { text: string }[];
    drivers_negative?: { text: string }[];
  };

  risk_drivers?: string[];
  protective_factors?: string[];
  top_features?: { feature: string; impact: number }[];

  final_decision?: string;
}

interface NeoScoreState {
  // Input
  profile: RawProfile;
  personaName: string | null;

  // Output
  result: ScoreResult | null;
  isLoading: boolean;
  error: string | null;

  // Actions
  setProfile: (profile: RawProfile, personaName?: string) => void;
  updateProfileField: <K extends keyof RawProfile>(key: K, value: RawProfile[K]) => void;
  // profileOverride: pass the profile directly to avoid race condition
  // where get().profile hasn't updated yet after setProfile().
  // All existing callers that pass no args continue to work unchanged.
  predict: (profileOverride?: RawProfile) => Promise<void>;
  clearError: () => void;
  reset: () => void;
}

// ─── Default profile ──────────────────────────────────────────────────────────

const DEFAULT_PROFILE: RawProfile = {
  AMT_INCOME_TOTAL: 180000,
  AMT_CREDIT: 300000,
  AMT_ANNUITY: 18000,
  AMT_GOODS_PRICE: 270000,
  AGE_YEARS: 30,
  CNT_CHILDREN: 0,
  CNT_FAM_MEMBERS: 2,
  EMPLOYED_YEARS: 3,
  NAME_INCOME_TYPE: "Working",
  OCCUPATION_TYPE: "Laborers",
  ORGANIZATION_TYPE: "Business Entity Type 3",
  NAME_EDUCATION_TYPE: "Secondary / secondary special",
  NAME_FAMILY_STATUS: "Married",
  FLAG_OWN_REALTY: 0,
  FLAG_OWN_CAR: 0,
  NAME_HOUSING_TYPE: "House / apartment",
  REGION_RATING_CLIENT: 2,
  REGION_POPULATION_RELATIVE: 0.02,
};

// ─── Backend URL ──────────────────────────────────────────────────────────────
// NEXT_PUBLIC_* vars are inlined at build time by Next.js.
// Must be set in Vercel dashboard → Settings → Environment Variables
// and a fresh deploy triggered for changes to take effect.
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000";

// ─── Store ────────────────────────────────────────────────────────────────────

export const useStore = create<NeoScoreState>((set, get) => ({
  profile: DEFAULT_PROFILE,
  personaName: null,
  result: null,
  isLoading: false,
  error: null,

  setProfile: (profile, personaName = undefined) =>
    set({ profile, personaName }),

  updateProfileField: (key, value) =>
    set((state) => ({
      profile: { ...state.profile, [key]: value },
    })),

  predict: async (profileOverride?: RawProfile) => {
    set({ isLoading: true, error: null });

    // Use override if provided — avoids race condition where store hasn't
    // updated yet when predict() is called right after setProfile().
    // Falls back to store profile for what-if simulator and other callers.
    const profile = profileOverride ?? get().profile;

    console.log("[NeoScore] Sending score request to:", API_BASE);
    console.log("[NeoScore] Profile being sent:", JSON.stringify(profile, null, 2));

    try {
      const res = await fetch(`${API_BASE}/api/score`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // No credentials:"include" — /api/score is a public endpoint.
        // Cross-origin cookies are blocked by browsers (SameSite=Lax).
        // credentials are only needed for /auth/* and /score/history.
        body: JSON.stringify({ features: profile }),
      });

      if (!res.ok) {
        let msg = `API error ${res.status}`;
        try {
          const body = await res.json();
          if (body?.error) msg = body.error;
        } catch {}
        throw new Error(msg);
      }

      const data: ScoreResult = await res.json();
      console.log("[NeoScore] Score received:", data.score, data.risk_tier);

      set({ result: data, isLoading: false, error: null });

      // Mirror to sessionStorage so results page survives a hard refresh
      try {
        sessionStorage.setItem("neoscore_result", JSON.stringify(data));
        sessionStorage.setItem("neoscore_features", JSON.stringify(profile));
      } catch {}

    } catch (err: any) {
      const isNetworkError =
        err?.message?.includes("fetch") ||
        err?.message?.includes("Failed to fetch") ||
        err?.message?.includes("NetworkError") ||
        err?.message?.includes("CORS");

      const message = isNetworkError
        ? `Cannot reach backend at ${API_BASE}. Check NEXT_PUBLIC_API_URL is set on Vercel and redeploy.`
        : err?.message ?? "Scoring failed";

      console.error("[NeoScore] predict() failed:", message);

      // Set result to null so results page redirects back to /home
      // instead of showing stale default data (score 300)
      set({ isLoading: false, error: message, result: null });

      throw new Error(message);
    }
  },

  clearError: () => set({ error: null }),

  reset: () =>
    set({
      profile: DEFAULT_PROFILE,
      personaName: null,
      result: null,
      isLoading: false,
      error: null,
    }),
}));
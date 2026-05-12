/**
 * useStore.ts — NeoScore global state
 *
 * Schema is aligned 1:1 with backend STUDENT_FEATURES.
 * No translation layer needed — profile fields ARE the API payload.
 *
 * FIXES:
 *  1. API_BASE now correctly reads NEXT_PUBLIC_API_URL at runtime
 *  2. Removed credentials:"include" from /api/score (public endpoint,
 *     cross-origin cookies are blocked by browsers anyway)
 *  3. On error, result is set to null so results page redirects instead
 *     of showing stale/default data (score 300)
 *  4. Better error message tells you exactly which URL failed
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
  predict: () => Promise<void>;
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
// before redeploying, otherwise this falls back to localhost.
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

  predict: async () => {
    set({ isLoading: true, error: null });

    const profile = get().profile;

    // Log which backend is being used — remove after confirming deployment works
    console.log("[NeoScore] Sending score request to:", API_BASE);

    try {
      const res = await fetch(`${API_BASE}/api/score`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Profile fields ARE the STUDENT_FEATURES — send directly.
        // NO credentials:"include" — /api/score is a public endpoint and
        // cross-origin cookies are blocked by browsers (SameSite=Lax).
        // credentials are only needed for /auth/* and /score/history routes.
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

      // Clear result first to avoid any flash of stale data
      set({ result: null });
      set({ result: data, isLoading: false, error: null });

      // Mirror to sessionStorage so results page survives a hard refresh
      try {
        sessionStorage.setItem("neoscore_result", JSON.stringify(data));
        sessionStorage.setItem("neoscore_features", JSON.stringify(profile));
      } catch {}

    } catch (err: any) {
      // Distinguish network failure (env var wrong) from API errors
      const isNetworkError =
        err?.message?.includes("fetch") ||
        err?.message?.includes("Failed to fetch") ||
        err?.message?.includes("NetworkError") ||
        err?.message?.includes("CORS");

      const message = isNetworkError
        ? `Cannot reach backend at ${API_BASE}. If deployed, check NEXT_PUBLIC_API_URL is set on Vercel and redeploy.`
        : err?.message ?? "Scoring failed";

      // Set result to null so the results page redirects back to /home
      // instead of showing stale default data (score 300)
      set({ isLoading: false, error: message, result: null });

      throw new Error(message); // re-throw so callers can catch
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
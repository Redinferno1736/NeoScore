/**
 * useStore.ts — NeoScore global state
 *
 * Schema is aligned 1:1 with backend STUDENT_FEATURES.
 * No translation layer needed — profile fields ARE the API payload.
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
// In production set NEXT_PUBLIC_API_URL in your .env
const API_BASE =
  typeof process !== "undefined"
    ? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000"
    : "http://localhost:5000";

// ─── Store ────────────────────────────────────────────────────────────────────

export const useStore = create<NeoScoreState>((set, get) => ({
  profile: DEFAULT_PROFILE,
  personaName: null,
  result: null,
  isLoading: false,
  error: null,

  setProfile: (profile, personaName = null) =>
    set({ profile, personaName }),

  updateProfileField: (key, value) =>
    set((state) => ({
      profile: { ...state.profile, [key]: value },
    })),

  predict: async () => {
    set({ isLoading: true, error: null });

    const profile = get().profile;

    try {
      const res = await fetch(`${API_BASE}/api/score`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Profile fields ARE the STUDENT_FEATURES — send directly
        body: JSON.stringify({ features: profile }),
        credentials: "include",
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
      set({ result: data, isLoading: false, error: null });

      // Mirror to sessionStorage so results page survives a hard refresh
      try {
        sessionStorage.setItem("neoscore_result", JSON.stringify(data));
        sessionStorage.setItem("neoscore_features", JSON.stringify(profile));
      } catch {}
    } catch (err: any) {
      const message =
        err?.message?.includes("fetch")
          ? "Cannot reach backend — is it running on port 5000?"
          : err?.message ?? "Scoring failed";
      set({ isLoading: false, error: message });
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
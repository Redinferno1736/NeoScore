import { create } from "zustand";
import { submitScore, fetchRecommendations } from "@/lib/api";

// ─── Types ────────────────────────────────────────────────────────────────────
export interface Profile {
  income: number;
  dti: number;
  savingsRatio: number;
  employmentDuration: string;
  age: number;
  ownsHouse: boolean;
  ownsCar: boolean;
  familySize: number;
  children: number;
  educationLevel: string;
}

interface StoreState {
  profile: Profile;
  result: any | null;
  isLoading: boolean;
  activePersona: string | null;
  updateProfileField: <K extends keyof Profile>(field: K, value: Profile[K]) => void;
  setProfile: (profile: Profile, personaName?: string) => void;
  predict: () => Promise<void>;
}

// ─── Human-readable feature labels ───────────────────────────────────────────
// Shown in the Score Drivers panel instead of raw snake_case column names.
const FEATURE_LABELS: Record<string, string> = {
  CNT_CHILDREN:              "Number of Children",
  CNT_FAM_MEMBERS:           "Family Size",
  NAME_EDUCATION_TYPE:       "Education Level",
  NAME_FAMILY_STATUS:        "Marital Status",
  NAME_INCOME_TYPE:          "Income Type",
  OCCUPATION_TYPE:           "Occupation",
  ORGANIZATION_TYPE:         "Employer Type",
  NAME_HOUSING_TYPE:         "Housing Situation",
  FLAG_OWN_CAR:              "Owns a Car",
  FLAG_OWN_REALTY:           "Owns Property",
  AMT_INCOME_TOTAL:          "Annual Income",
  AMT_CREDIT:                "Loan Amount",
  AMT_ANNUITY:               "Monthly EMI",
  AMT_GOODS_PRICE:           "Goods Price",
  REGION_POPULATION_RELATIVE:"Region Population Density",
  REGION_RATING_CLIENT:      "Region Credit Rating",
  AGE_YEARS:                 "Age",
  EMPLOYED_YEARS:            "Years Employed",
  DEBT_TO_INCOME:            "Debt-to-Income Ratio",
  ANNUITY_TO_INCOME:         "Annuity-to-Income Ratio",
  CREDIT_TO_GOODS:           "Credit-to-Goods Ratio",
  INCOME_PER_PERSON:         "Income per Family Member",
  CHILDREN_RATIO:            "Children Ratio",
  EMI_BURDEN:                "EMI Burden",
  INCOME_STABILITY:          "Income Stability",
  LOAN_TO_INCOME:            "Loan-to-Income Ratio",
  FINANCIAL_PRESSURE:        "Financial Pressure",
  STABILITY_SCORE:           "Stability Score",
  ASSET_SCORE:               "Asset Score",
  INCOME_ADEQUACY:           "Income Adequacy",
};

// ─── Education dropdown → Home Credit class name translation ──────────────────
// Must mirror FRONTEND_VALUE_MAP in backend/main.py exactly.
const EDUCATION_MAP: Record<string, string> = {
  "High School": "Secondary / secondary special",
  "Bachelor":    "Higher education",
  "Master":      "Higher education",
  "PhD":         "Academic degree",
};

// ─── Default profile ──────────────────────────────────────────────────────────
const defaultProfile: Profile = {
  income:             50000,
  dti:                20,
  savingsRatio:       10,
  employmentDuration: "1-3 years",
  age:                28,
  ownsHouse:          false,
  ownsCar:            false,
  familySize:         1,
  children:           0,
  educationLevel:     "Bachelor",
};

// ─── SHAP impact → display points conversion ──────────────────────────────────
// SHAP values for XGBRegressor (binary:logistic) are in log-odds space,
// typically ranging from about -2 to +2 for strong features.
// We convert to an approximate score-point impact using the score range (600pts).
// A SHAP of ±0.1 log-odds ≈ ±2.5% probability change ≈ ~15 score points.
function shapToPoints(shapVal: number): number {
  // Scale factor: empirically tuned so typical SHAP values map to 5–50 pts.
  // Adjust SHAP_SCALE if your model's values are systematically larger/smaller.
  const SHAP_SCALE = 150;
  return Math.max(1, Math.round(Math.abs(shapVal) * SHAP_SCALE));
}

// ─── Store ────────────────────────────────────────────────────────────────────
export const useStore = create<StoreState>((set, get) => ({
  profile:      defaultProfile,
  result:       null,
  isLoading:    false,
  activePersona: null,

  updateProfileField: (field, value) =>
    set((state) => ({ profile: { ...state.profile, [field]: value } })),

  setProfile: (profile, personaName) =>
    set({ profile, activePersona: personaName || null }),

  predict: async () => {
    set({ isLoading: true });
    const { profile } = get();

    // ── Map frontend profile → backend feature names ──────────────────────────
    // NAME_EDUCATION_TYPE: translate simplified label → Home Credit class name
    const educationValue =
      EDUCATION_MAP[profile.educationLevel] ?? profile.educationLevel;

    const backendPayload: Record<string, any> = {
      AMT_INCOME_TOTAL:    profile.income,
      // DTI from the slider is a percentage (e.g. 15 = 15%).
      // The model expects the raw ratio (credit / income), not a percentage.
      // We approximate: if dti% is the annuity-to-income ratio, then
      // AMT_CREDIT ≈ income * dti/100 * 12  (annualised)
      AMT_CREDIT:          profile.income * (profile.dti / 100) * 12,
      AMT_ANNUITY:         profile.income * (profile.dti / 100),
      AMT_GOODS_PRICE:     profile.income * (profile.dti / 100) * 11, // ~LTV of 0.92
      AGE_YEARS:           profile.age,
      FLAG_OWN_REALTY:     profile.ownsHouse ? 1 : 0,
      FLAG_OWN_CAR:        profile.ownsCar   ? 1 : 0,
      CNT_FAM_MEMBERS:     profile.familySize,
      CNT_CHILDREN:        profile.children,
      // Send the translated education string; backend will label-encode it
      NAME_EDUCATION_TYPE: educationValue,
      EMPLOYED_YEARS:
        profile.employmentDuration === "< 1 year"  ? 0.5
        : profile.employmentDuration === "1-3 years" ? 2
        : profile.employmentDuration === "3-5 years" ? 4
        : 6,
    };

    try {
      // 1. Score + SHAP from Flask
      const apiResponse = await submitScore(backendPayload);

      // 2. Counterfactual recommendations
      let aiRecommendations: any[] = [];
      try {
        const cfResponse = await fetchRecommendations(backendPayload);
        if (cfResponse.moves) {
          aiRecommendations = cfResponse.moves.map((move: any) => ({
            action: move.label,
            effort: move.effort.charAt(0).toUpperCase() + move.effort.slice(1),
            impact: Math.round(move.score_delta),
          }));
        }
      } catch (cfErr) {
        console.warn("Counterfactual engine unavailable.", cfErr);
      }

      // 3. Map API response → UI shape
      //    top_features from the API: [{ feature: string, impact: float (SHAP log-odds) }]
      //    In binary:logistic XGBRegressor:
      //      POSITIVE SHAP → increases predicted default probability → HURTS score → red
      //      NEGATIVE SHAP → decreases predicted default probability → HELPS score → green
      const featureImpacts = (apiResponse.top_features ?? []).map((f: any) => ({
        feature:  FEATURE_LABELS[f.feature] ?? f.feature,
        impact:   shapToPoints(f.impact),
        // negative SHAP = helps score = show as positive (green)
        positive: f.impact < 0,
      }));

      const mappedResult = {
        score:   apiResponse.score,
        maxScore: 900,
        risk:
          apiResponse.risk_tier === "Excellent" || apiResponse.risk_tier === "Good"
            ? "Low"
            : apiResponse.risk_tier === "Fair"
            ? "Medium"
            : "High",
        coachMessage: apiResponse.reasoning,
        featureImpacts,
        recommendations:
          aiRecommendations.length > 0
            ? aiRecommendations
            : [
                { action: "Decrease Debt-to-Income by 5%",        effort: "Medium", impact: 25 },
                { action: "Increase Savings Ratio to 20%",         effort: "High",   impact: 40 },
                { action: "Maintain current employment for 1 yr",  effort: "Low",    impact: 15 },
              ],
      };

      set({ result: mappedResult, isLoading: false });
    } catch (error) {
      console.error("Failed to predict score:", error);
      set({ isLoading: false });
    }
  },
}));

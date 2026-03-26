import { create } from "zustand";
import { submitScore, fetchRecommendations } from "@/lib/api";

// 1. Define the TypeScript interfaces
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
  result: any | null; // Changed from scoreData to result to match your UI
  isLoading: boolean; // Added loading state for your spinner and circle animation
  activePersona: string | null;
  updateProfileField: <K extends keyof Profile>(field: K, value: Profile[K]) => void;
  setProfile: (profile: Profile, personaName?: string) => void;
  predict: () => Promise<void>;
}

// 2. Set a default blank state
const defaultProfile: Profile = {
  income: 50000,
  dti: 20,
  savingsRatio: 10,
  employmentDuration: "1-3 years",
  age: 28,
  ownsHouse: false,
  ownsCar: false,
  familySize: 1,
  children: 0,
  educationLevel: "Bachelor",
};

// 3. Create the global store
export const useStore = create<StoreState>((set, get) => ({
  profile: defaultProfile,
  result: null, 
  isLoading: false, 
  activePersona: null,

  updateProfileField: (field, value) =>
    set((state) => ({
      profile: { ...state.profile, [field]: value },
    })),

  setProfile: (profile, personaName) =>
    set({ profile, activePersona: personaName || null }),

  predict: async () => {
    set({ isLoading: true }); // Start the loading spinner on the UI
    const { profile } = get();

    // Map React state to Flask backend column names
    const backendPayload = {
      AMT_INCOME_TOTAL: profile.income,
      DEBT_TO_INCOME: profile.dti / 100, 
      AGE_YEARS: profile.age,
      FLAG_OWN_REALTY: profile.ownsHouse ? 1 : 0,
      FLAG_OWN_CAR: profile.ownsCar ? 1 : 0,
      CNT_FAM_MEMBERS: profile.familySize,
      CNT_CHILDREN: profile.children,
      NAME_EDUCATION_TYPE: profile.educationLevel,
      EMPLOYED_YEARS:
        profile.employmentDuration === "< 1 year" ? 0.5
          : profile.employmentDuration === "1-3 years" ? 2
          : profile.employmentDuration === "3-5 years" ? 4
          : 6,
    };

    try {
      // 1. Get the real score and SHAP data from Flask
      const apiResponse = await submitScore(backendPayload);

      // 2. Ask the AI Counterfactual Engine how to improve this specific score
      let aiRecommendations = [];
      try {
          const cfResponse = await fetchRecommendations(backendPayload);
          // Map the Python response ('moves') to your React UI format
          if (cfResponse.moves) {
              aiRecommendations = cfResponse.moves.map((move: any) => ({
                  action: move.label,
                  // Ensure formatting matches your green/yellow/red UI tags
                  effort: move.effort.charAt(0).toUpperCase() + move.effort.slice(1), 
                  impact: Math.round(move.score_delta)
              }));
          }
      } catch (cfError) {
          console.warn("Counterfactual engine unavailable, using fallbacks.", cfError);
      }

      // 3. Map everything exactly as your ResultsPage.tsx expects
      const mappedResult = {
        score: apiResponse.score,
        maxScore: 900,
        risk: apiResponse.risk_tier === "Excellent" || apiResponse.risk_tier === "Good" ? "Low" 
            : apiResponse.risk_tier === "Fair" ? "Medium" 
            : "High",
        coachMessage: apiResponse.reasoning, 
        
        featureImpacts: apiResponse.top_features.map((f: any) => ({
          feature: f.feature,
          impact: Math.round(Math.abs(f.impact * 10)), 
          positive: f.impact > 0
        })),

        // 4. Inject the REAL AI recommendations (or fallbacks if it failed)
        recommendations: aiRecommendations.length > 0 ? aiRecommendations : [
          { action: "Decrease Debt-to-Income by 5%", effort: "Medium", impact: 25 },
          { action: "Increase Savings Ratio to 20%", effort: "High", impact: 40 },
          { action: "Maintain current employment for 1 yr", effort: "Low", impact: 15 }
        ]
      };

      set({ result: mappedResult, isLoading: false });

    } catch (error) {
      console.error("Failed to predict score:", error);
      set({ isLoading: false }); 
    }
  },
}));
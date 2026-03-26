import { create } from 'zustand';
import { persist } from 'zustand/middleware';

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

export interface Recommendation {
  action: string;
  effort: 'Low' | 'Medium' | 'High';
  impact: number;
}

export interface FeatureImpact {
  feature: string;
  impact: number;
  positive: boolean;
}

export interface Result {
  score: number;
  maxScore: number;
  risk: 'Low' | 'Medium' | 'High';
  featureImpacts: FeatureImpact[];
  recommendations: Recommendation[];
  coachMessage: string;
}

export interface HistoryEntry {
  id: string;
  timestamp: string;
  personaName: string | null;
  profile: Profile;
  result: Result;
}

interface StoreState {
  profile: Profile;
  result: Result | null;
  history: HistoryEntry[];
  selectedPersona: string | null;
  isLoading: boolean;
  
  updateProfileField: <K extends keyof Profile>(field: K, value: Profile[K]) => void;
  setProfile: (profile: Profile, personaName?: string | null) => void;
  predict: () => Promise<void>;
  deleteHistoryEntry: (id: string) => void;
  clearHistory: () => void;
}

export const defaultProfile: Profile = {
  income: 60000,
  dti: 30,
  savingsRatio: 20,
  employmentDuration: '1-3 years',
  age: 30,
  ownsHouse: false,
  ownsCar: false,
  familySize: 1,
  children: 0,
  educationLevel: 'Bachelor',
};

// Mock prediction logic
const mockPredict = async (profile: Profile): Promise<Result> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      // Very basic Mock ML logic for demo purposes
      let score = 500;
      score += (profile.income / 1000) * 1.5;
      score -= profile.dti * 2.5;
      score += profile.savingsRatio * 3;
      if (profile.ownsHouse) score += 40;
      if (profile.ownsCar) score += 15;
      if (profile.educationLevel === 'PhD' || profile.educationLevel === 'Master') score += 20;

      score = Math.min(Math.max(Math.round(score), 300), 900);
      
      let risk: 'Low' | 'Medium' | 'High' = 'Medium';
      if (score > 720) risk = 'Low';
      if (score < 580) risk = 'High';

      resolve({
        score,
        maxScore: Math.min(score + 140, 900),
        risk,
        featureImpacts: [
          { feature: 'Income Level', impact: Math.floor(profile.income / 2500), positive: true },
          { feature: 'Debt-to-Income', impact: Math.floor(profile.dti * 1.5), positive: false },
          { feature: 'Savings Habit', impact: Math.floor(profile.savingsRatio * 2), positive: true },
        ],
        recommendations: [
          { action: 'Pay down credit card balances', effort: 'Medium', impact: 45 },
          { action: 'Increase monthly savings contribution', effort: 'Low', impact: 15 },
          { action: 'Avoid new credit inquiries for 6 months', effort: 'High', impact: 10 },
        ],
        coachMessage: `You are in the ${risk.toLowerCase()} risk category. Focusing on keeping your debt-to-income ratio low while increasing your savings will yield the best improvements in your score.`,
      });
    }, 600); // 600ms network simulated delay
  });
};

export const useStore = create<StoreState>()(
  persist(
    (set, get) => ({
      profile: defaultProfile,
      result: null,
      history: [],
      selectedPersona: null,
      isLoading: false,

      updateProfileField: (field, value) => {
        set((state) => ({
          profile: { ...state.profile, [field]: value },
          selectedPersona: null, // manual override drops persona tag
        }));
      },

      setProfile: (profile, personaName = null) => {
        set({ profile, selectedPersona: personaName });
      },

      predict: async () => {
        set({ isLoading: true });
        const { profile, selectedPersona } = get();
        const result = await mockPredict(profile);
        
        const historyEntry: HistoryEntry = {
          id: Date.now().toString(),
          timestamp: new Date().toISOString(),
          personaName: selectedPersona,
          profile,
          result,
        };

        set((state) => ({
          result,
          isLoading: false,
          history: [historyEntry, ...state.history],
        }));
      },

      deleteHistoryEntry: (id) => set((state) => ({
        history: state.history.filter((entry) => entry.id !== id),
      })),

      clearHistory: () => set({ history: [] }),
    }),
    {
      name: 'neoscore-global-store',
      // In a real app we might only persist history, but persisting profile and result makes the back button seamless.
    }
  )
);

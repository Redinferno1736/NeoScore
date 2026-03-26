// frontend/store/useScoreStore.ts
import { create } from 'zustand';

interface ScoreState {
    scoreData: any | null;
    setScoreData: (data: any) => void;
    clearScoreData: () => void;
}

export const useScoreStore = create<ScoreState>((set) => ({
    scoreData: null,
    setScoreData: (data) => set({ scoreData: data }),
    clearScoreData: () => set({ scoreData: null }),
}));
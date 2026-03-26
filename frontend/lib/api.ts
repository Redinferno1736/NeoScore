// frontend/lib/api.ts
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";

const fetchConfig: RequestInit = {
    headers: { "Content-Type": "application/json" },
    credentials: "include", // CRITICAL: Sends session cookies to Flask
};

export async function checkAuth() {
    const res = await fetch(`${API_BASE}/auth/me`, { ...fetchConfig, method: "GET" });
    if (!res.ok) throw new Error("Auth check failed");
    return res.json();
}

export async function submitScore(features: Record<string, any>) {
    const targetUrl = `${API_BASE}/api/score`;
    console.log("Attempting to fetch from:", targetUrl); // ADD THIS LINE
    
    const res = await fetch(targetUrl, {
        ...fetchConfig,
        method: "POST",
        body: JSON.stringify({ features }),
    });
    
    if (!res.ok) throw new Error("Failed to calculate score");
    return res.json();
}


export async function fetchRecommendations(features: Record<string, any>) {
    const res = await fetch(`${API_BASE}/api/counterfactual`, {
        ...fetchConfig,
        method: "POST",
        body: JSON.stringify({ features }),
    });
    
    if (!res.ok) throw new Error("Failed to fetch counterfactual recommendations");
    return res.json();
}
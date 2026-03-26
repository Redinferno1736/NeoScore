import { Card } from "@/components/ui/Card";
import { Terminal, Code, ServerCrash, Copy } from "lucide-react";

export default function ApiDocsPage() {
  const MOCK_RESPONSE = {
    score: 720,
    riskLevel: "Low",
    featureImpacts: [
      { feature: "Income Level", impact: 15, positive: true },
      { feature: "Debt-to-Income", impact: -20, positive: false }
    ]
  };

  const MOCK_REQUEST = {
    income: 85000,
    dti: 15,
    savingsRatio: 25,
    employmentDuration: "> 5 years",
    age: 34,
    ownsHouse: true,
    ownsCar: true,
    familySize: 2,
    children: 0,
    educationLevel: "Master"
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 pb-16">
      <div className="mt-2 border-b border-border pb-4">
        <h1 className="text-3xl font-bold flex items-center gap-3 text-text">
          <Terminal className="text-accent" /> API Reference
        </h1>
        <p className="text-muted text-sm mt-1">
          Integrate the NeoScore prediction engine directly into your own applications.
        </p>
      </div>

      <section className="space-y-6">
        <div>
          <h2 className="text-2xl font-semibold mb-4 text-accent flex items-center gap-2">
            <ServerCrash size={20} /> Endpoint: POST /predict
          </h2>
          <Card className="p-6 bg-card border-border shadow-md">
            <p className="text-muted text-sm leading-relaxed mb-6">
              The prediction endpoint evaluates a user&apos;s financial profile using our proprietary ML model
              to return an explainable credit score, risk tier, and impact factors.
            </p>

            <div className="space-y-4">
              <h3 className="font-bold text-lg mb-2 flex items-center gap-2 text-text"><Code size={18}/> Request Payload</h3>
              <div className="relative group">
                <button className="absolute top-3 right-3 p-1.5 rounded-lg bg-bg text-muted hover:text-accent border border-border transition-colors shadow-sm">
                  <Copy size={14} />
                </button>
                <pre className="bg-bg border border-border rounded-xl p-4 text-sm overflow-x-auto text-green-400 font-mono">
                  {JSON.stringify(MOCK_REQUEST, null, 2)}
                </pre>
              </div>

              <h3 className="font-bold text-lg mb-2 flex items-center gap-2 pt-4 text-text"><Code size={18}/> Response payload</h3>
              <div className="relative group">
                <button className="absolute top-3 right-3 p-1.5 rounded-lg bg-bg text-muted hover:text-accent border border-border transition-colors shadow-sm">
                  <Copy size={14} />
                </button>
                <pre className="bg-bg border border-border rounded-xl p-4 text-sm overflow-x-auto text-blue-400 font-mono">
                  {JSON.stringify(MOCK_RESPONSE, null, 2)}
                </pre>
              </div>
            </div>
          </Card>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4">
          <Card className="p-6">
            <h3 className="font-bold mb-4 flex items-center gap-2 text-text">
              <Terminal size={16} className="text-accent"/> JavaScript (Fetch)
            </h3>
            <pre className="bg-bg border border-border rounded-xl p-4 text-xs overflow-x-auto text-orange-300 font-mono shadow-inner">
{`const response = await fetch('https://api.neoscore.dev/v1/predict', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(profileData)
});

const data = await response.json();
console.log(data.score, data.riskLevel);`}
            </pre>
          </Card>
          
          <Card className="p-6">
            <h3 className="font-bold mb-4 flex items-center gap-2 text-text">
              <Terminal size={16} className="text-accent"/> Python (Requests)
            </h3>
            <pre className="bg-bg border border-border rounded-xl p-4 text-xs overflow-x-auto text-yellow-300 font-mono shadow-inner">
{`import requests

url = "https://api.neoscore.dev/v1/predict"
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

response = requests.post(
    url, 
    json=profile_data, 
    headers=headers
)
data = response.json()
print(f"Score: {data['score']}")`}
            </pre>
          </Card>
        </div>
      </section>
    </div>
  );
}

import { Card } from "@/components/ui/Card";
import { Terminal, Code, ServerCrash, Copy, Zap, Shield, Activity } from "lucide-react";

const CODE = {
  scoreRequest: JSON.stringify({
    features: {
      AMT_INCOME_TOTAL: 300000,
      AMT_CREDIT: 500000,
      AMT_ANNUITY: 25000,
      AGE_YEARS: 32,
      EMPLOYED_YEARS: 5,
      FLAG_OWN_REALTY: 1,
      FLAG_OWN_CAR: 0,
      CNT_FAM_MEMBERS: 2,
      CNT_CHILDREN: 0,
      NAME_EDUCATION_TYPE: "Higher education",
      NAME_INCOME_TYPE: "Working",
      NAME_HOUSING_TYPE: "House / apartment",
      NAME_FAMILY_STATUS: "Married",
    },
  }, null, 2),

  scoreResponse: JSON.stringify({
    score: 724,
    risk_tier: "Good",
    default_probability: 0.1037,
    approval_probability: 0.8963,
    percentile: 71.4,
    reasoning: "Your employment stability is working in your favour, but your loan-to-income ratio is the biggest drag on your score.",
    top_features: [
      { feature: "EMPLOYED_YEARS",   impact: -0.042 },
      { feature: "LOAN_TO_INCOME",   impact:  0.031 },
      { feature: "AMT_INCOME_TOTAL", impact: -0.027 },
      { feature: "ASSET_SCORE",      impact: -0.019 },
      { feature: "EMI_BURDEN",       impact:  0.015 },
    ],
  }, null, 2),

  cfRequest: JSON.stringify({
    features: {
      AMT_INCOME_TOTAL: 120000,
      AMT_CREDIT: 100000,
      EMPLOYED_YEARS: 0.8,
      AGE_YEARS: 28,
      FLAG_OWN_REALTY: 0,
    },
    max_steps: 3,
  }, null, 2),

  cfResponse: JSON.stringify({
    original_score: 551,
    original_risk: "Very Poor",
    original_probability: 0.5817,
    best_reachable_score: 589,
    moves: [
      { label: "Job tenure",            score_delta: 28, effort: "low",    timeline_days: 90  },
      { label: "Annual income",         score_delta: 11, effort: "medium", timeline_days: 90  },
      { label: "Loan amount requested", score_delta:  9, effort: "low",    timeline_days: 0   },
    ],
  }, null, 2),

  personaRequest: JSON.stringify({ persona: "priya" }, null, 2),

  js: `const response = await fetch('http://localhost:5000/api/score', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    features: {
      AMT_INCOME_TOTAL: 300000,
      EMPLOYED_YEARS: 5,
      NAME_EDUCATION_TYPE: 'Higher education',
      // ... other features (missing ones use smart defaults)
    }
  })
});
const data = await response.json();
console.log(data.score, data.risk_tier);`,

  python: `import requests

response = requests.post(
    "http://localhost:5000/api/score",
    json={
        "features": {
            "AMT_INCOME_TOTAL": 300000,
            "EMPLOYED_YEARS": 5,
            "NAME_EDUCATION_TYPE": "Higher education",
            # missing features use smart defaults
        }
    }
)
data = response.json()
print(f"Score: {data['score']} | Tier: {data['risk_tier']}")`,
};

function CodeBlock({ code, lang }: { code: string; lang: string }) {
  const colorMap: Record<string, string> = {
    json: "text-green-400",
    js: "text-orange-300",
    python: "text-yellow-300",
  };
  return (
    <div className="relative group">
      <span className="absolute top-2 right-8 text-[10px] uppercase tracking-widest text-muted">
        {lang}
      </span>
      <button className="absolute top-2 right-2 p-1 rounded text-muted hover:text-accent transition-colors">
        <Copy size={13} />
      </button>
      <pre
        className={`bg-bg border border-border rounded-xl p-4 text-xs overflow-x-auto font-mono leading-relaxed ${colorMap[lang] ?? "text-text"}`}
      >
        {code}
      </pre>
    </div>
  );
}

function EndpointBadge({ method, path }: { method: string; path: string }) {
  return (
    <div className="flex items-center gap-3 mb-4">
      <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-accent/10 text-accent border border-accent/20 font-mono">
        {method}
      </span>
      <code className="text-sm text-text font-mono">{path}</code>
    </div>
  );
}

export default function ApiDocsPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 space-y-10 pb-16 pt-8">
      {/* Header */}
      <div className="border-b border-border pb-6">
        <h1 className="text-3xl font-bold flex items-center gap-3 text-text font-serif">
          <Terminal className="text-accent" /> API Reference
        </h1>
        <p className="text-muted text-sm mt-2 max-w-xl">
          The NeoScore scoring engine is accessible as a local REST API at{" "}
          <code className="text-accent bg-accent/10 px-1 rounded">http://localhost:5000</code>.
          No API key required for public endpoints.
        </p>

        {/* Feature pills */}
        <div className="flex flex-wrap gap-3 mt-4">
          {[
            { icon: Zap,      label: "No auth for /api/* routes" },
            { icon: Shield,   label: "Session auth for /score/* routes" },
            { icon: Activity, label: "SHAP explanations included" },
          ].map(({ icon: Icon, label }) => (
            <div
              key={label}
              className="flex items-center gap-1.5 text-xs text-muted border border-border rounded-full px-3 py-1"
            >
              <Icon size={12} className="text-accent" /> {label}
            </div>
          ))}
        </div>
      </div>

      {/* Score endpoint */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-accent flex items-center gap-2">
          <ServerCrash size={18} /> Scoring
        </h2>
        <Card className="p-6 space-y-5">
          <EndpointBadge method="POST" path="/api/score" />
          <p className="text-sm text-muted leading-relaxed">
            Pass any subset of the 30 student features. Missing fields are filled
            with population-median defaults. Returns a score (300–900), risk tier,
            default probability, percentile rank, SHAP-based top drivers, and a
            plain-English coaching message.
          </p>
          <p className="text-xs text-muted">
            Shortcut: pass{" "}
            <code className="text-accent bg-accent/10 px-1 rounded">
              {`{ "persona": "ravi" }`}
            </code>{" "}
            to use a built-in demo profile (ravi · priya · deepa).
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-1">
              <p className="text-xs font-semibold text-muted uppercase tracking-widest">Request</p>
              <CodeBlock code={CODE.scoreRequest} lang="json" />
            </div>
            <div className="space-y-1">
              <p className="text-xs font-semibold text-muted uppercase tracking-widest">Response</p>
              <CodeBlock code={CODE.scoreResponse} lang="json" />
            </div>
          </div>
        </Card>
      </section>

      {/* Counterfactual endpoint */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-accent flex items-center gap-2">
          <Code size={18} /> Counterfactual Recommendations
        </h2>
        <Card className="p-6 space-y-5">
          <EndpointBadge method="POST" path="/api/counterfactual" />
          <p className="text-sm text-muted leading-relaxed">
            Given a feature profile, returns the top actionable steps that would
            increase the score the most. Immutable features (age, family size,
            region) are automatically excluded.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-1">
              <p className="text-xs font-semibold text-muted uppercase tracking-widest">Request</p>
              <CodeBlock code={CODE.cfRequest} lang="json" />
            </div>
            <div className="space-y-1">
              <p className="text-xs font-semibold text-muted uppercase tracking-widest">Response</p>
              <CodeBlock code={CODE.cfResponse} lang="json" />
            </div>
          </div>
        </Card>
      </section>

      {/* Simulate endpoint */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-accent flex items-center gap-2">
          <Activity size={18} /> What-If Simulation
        </h2>
        <Card className="p-6 space-y-3">
          <EndpointBadge method="POST" path="/api/simulate" />
          <p className="text-sm text-muted leading-relaxed">
            Provide a base feature set and a{" "}
            <code className="text-accent bg-accent/10 px-1 rounded">changes</code> dict.
            Returns both the original and new score so you can measure the exact impact
            of any hypothetical change.
          </p>
          <CodeBlock
            lang="json"
            code={JSON.stringify({
              features: { AMT_INCOME_TOTAL: 120000, EMPLOYED_YEARS: 1 },
              changes:  { EMPLOYED_YEARS: 3, AMT_INCOME_TOTAL: 150000 },
            }, null, 2)}
          />
        </Card>
      </section>

      {/* Personas */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-accent flex items-center gap-2">
          <Zap size={18} /> Demo Personas
        </h2>
        <Card className="p-6 space-y-3">
          <p className="text-sm text-muted">
            All three scoring endpoints accept a{" "}
            <code className="text-accent bg-accent/10 px-1 rounded">persona</code> key
            as a shortcut:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            {[
              { key: "ravi",  label: "Ravi",  desc: "Gig worker, thin file, low income" },
              { key: "priya", label: "Priya", desc: "Salaried professional, good profile" },
              { key: "deepa", label: "Deepa", desc: "Self-employed, moderate risk" },
            ].map((p) => (
              <div key={p.key} className="border border-border rounded-lg p-3 space-y-1">
                <div className="font-semibold text-text">{p.label}</div>
                <div className="text-xs text-muted">{p.desc}</div>
                <code className="text-[11px] text-accent block mt-2">
                  {`{ "persona": "${p.key}" }`}
                </code>
              </div>
            ))}
          </div>
        </Card>
      </section>

      {/* Code samples */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-accent flex items-center gap-2">
          <Terminal size={18} /> Code Examples
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="p-5 space-y-2">
            <p className="text-xs font-semibold text-muted uppercase tracking-widest">JavaScript</p>
            <CodeBlock code={CODE.js} lang="js" />
          </Card>
          <Card className="p-5 space-y-2">
            <p className="text-xs font-semibold text-muted uppercase tracking-widest">Python</p>
            <CodeBlock code={CODE.python} lang="python" />
          </Card>
        </div>
      </section>

      {/* Feature reference */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-accent flex items-center gap-2">
          <Shield size={18} /> Feature Reference
        </h2>
        <Card className="p-6">
          <p className="text-sm text-muted mb-4">
            All 30 student features the model accepts. All are optional — missing values
            are filled with population medians.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-1 text-xs font-mono">
            {[
              ["CNT_CHILDREN",              "int",    "Number of children"],
              ["CNT_FAM_MEMBERS",           "int",    "Family size"],
              ["NAME_EDUCATION_TYPE",       "string", "Education level (Home Credit classes)"],
              ["NAME_FAMILY_STATUS",        "string", "Marital status"],
              ["NAME_INCOME_TYPE",          "string", "Income type"],
              ["OCCUPATION_TYPE",           "string", "Occupation"],
              ["ORGANIZATION_TYPE",         "string", "Employer type"],
              ["NAME_HOUSING_TYPE",         "string", "Housing situation"],
              ["FLAG_OWN_CAR",              "0/1",    "Owns a vehicle"],
              ["FLAG_OWN_REALTY",           "0/1",    "Owns property"],
              ["AMT_INCOME_TOTAL",          "float",  "Annual income (₹)"],
              ["AMT_CREDIT",               "float",  "Loan amount requested (₹)"],
              ["AMT_ANNUITY",              "float",  "Monthly EMI (₹)"],
              ["AMT_GOODS_PRICE",          "float",  "Goods price (₹)"],
              ["REGION_POPULATION_RELATIVE","float", "Region population density"],
              ["REGION_RATING_CLIENT",      "1-3",   "Region credit rating"],
              ["AGE_YEARS",                "float",  "Age in years"],
              ["EMPLOYED_YEARS",           "float",  "Years at current employer"],
            ].map(([name, type, desc]) => (
              <div key={name} className="flex gap-2 py-1 border-b border-border/40">
                <span className="text-accent w-44 shrink-0">{name}</span>
                <span className="text-muted w-12 shrink-0">{type}</span>
                <span className="text-muted/70">{desc}</span>
              </div>
            ))}
          </div>
          <p className="text-xs text-muted mt-4">
            + 12 derived ratio features auto-computed from the above (DEBT_TO_INCOME, EMI_BURDEN, etc.)
          </p>
        </Card>
      </section>
    </div>
  );
}

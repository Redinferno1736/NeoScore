import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { CheckCircle2, Building, DollarSign, Percent } from "lucide-react";

export default function LoansPage() {
  const MOCK_LOANS = [
    {
      id: 1,
      lender: "NeoCapital",
      type: "Personal Loan",
      amountRange: "$5,000 - $50,000",
      rateRange: "5.99% - 14.99%",
      term: "36 - 60 mo",
      eligibilityMatch: "High",
      features: ["No origination fees", "Next-day funding", "Autopay discount"],
    },
    {
      id: 2,
      lender: "VaultFlow Finance",
      type: "Debt Consolidation",
      amountRange: "$10,000 - $100,000",
      rateRange: "6.50% - 18.00%",
      term: "12 - 72 mo",
      eligibilityMatch: "Very High",
      features: ["Direct pay to creditors", "Soft credit pull", "Co-signers allowed"],
    },
    {
      id: 3,
      lender: "Apex Lending",
      type: "Home Improvement",
      amountRange: "$20,000 - $150,000",
      rateRange: "7.25% - 21.00%",
      term: "24 - 120 mo",
      eligibilityMatch: "Medium",
      features: ["Large load limits", "Secured options", "Flexible terms"],
    },
    {
      id: 4,
      lender: "GoldStandard Bank",
      type: "Auto Refinance",
      amountRange: "$15,000 - $80,000",
      rateRange: "4.99% - 12.99%",
      term: "36 - 84 mo",
      eligibilityMatch: "High",
      features: ["100% online process", "No vehicle age limit", "Skip a payment"],
    },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8 pb-16">
      <div className="mt-2 border-b border-border pb-4">
        <h1 className="text-3xl font-bold flex items-center gap-3 text-text">
          <DollarSign className="text-accent" /> Premium Loan Options
        </h1>
        <p className="text-muted text-sm mt-1">
          Tailored financial products matched to your precise credit profile.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {MOCK_LOANS.map((loan) => (
          <Card key={loan.id} className="p-6 hover:border-accent/30 transition-all flex flex-col group relative overflow-hidden">
            {loan.eligibilityMatch.includes("High") && (
              <div className="absolute top-0 right-0 py-1 cursor-default px-6 bg-green-500/10 text-green-500 font-bold text-xs uppercase tracking-wider transform translate-x-[30%] translate-y-2 rotate-45 z-10 w-40 text-center shadow-lg border border-green-500/20">
                Top Match
              </div>
            )}
            
            <div className="flex items-center gap-4 mb-6">
              <div className="h-12 w-12 rounded-xl bg-card flex items-center justify-center border border-border group-hover:border-accent/40 shadow-sm transition-colors">
                <Building className="text-accent" size={24} />
              </div>
              <div>
                <h3 className="text-xl font-bold leading-none mb-1 text-text">{loan.lender}</h3>
                <span className="text-sm text-muted">{loan.type}</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-6">
               <div className="bg-bg/50 rounded-xl p-3 border border-border">
                 <span className="block text-xs text-muted mb-1 flex items-center gap-1"><Percent size={12}/> Est. APR</span>
                 <span className="font-bold text-text text-sm">{loan.rateRange}</span>
               </div>
               <div className="bg-bg/50 rounded-xl p-3 border border-border">
                 <span className="block text-xs text-muted mb-1 flex items-center gap-1"><DollarSign size={12}/> Amount</span>
                 <span className="font-bold text-text text-sm">{loan.amountRange}</span>
               </div>
            </div>

            <div className="space-y-2 mb-8 flex-1">
               {loan.features.map((feature, i) => (
                 <div key={i} className="flex items-center gap-2 text-sm text-muted">
                    <CheckCircle2 size={16} className="text-accent" />
                    <span className="text-text">{feature}</span>
                 </div>
               ))}
            </div>

            <div className="pt-4 border-t border-border flex gap-3">
              <Button className="flex-1" variant="accent">
                Check Rates
              </Button>
              <Button className="flex-1" variant="outline">
                Details
              </Button>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

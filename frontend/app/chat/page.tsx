"use client";

import { useState, useRef, useEffect } from "react";
import { useStore } from "@/store/useStore";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { motion, AnimatePresence } from "framer-motion";
import { BrainCircuit, Send, User, Loader2, Sparkles } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const STARTER_PROMPTS = [
  "What does my credit score of mean for loan eligibility?",
  "How can I improve my EMI burden?",
  "What is debt-to-income ratio and why does it matter?",
  "Explain income stability in simple terms.",
  "What's the fastest way to raise my NeoScore?",
];

export default function ChatPage() {
  const { result } = useStore();
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hi! I'm your NeoScore AI financial coach. I can help you understand your credit score, explain what each factor means, and give you personalised advice on improving your creditworthiness. What would you like to know?",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || isLoading) return;

    const userMsg: Message = { role: "user", content: trimmed };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const history = messages.slice(-10); // last 10 turns for context
      const scoreContext = result
        ? {
            score: result.score,
            risk: result.risk,
            top_factors: result.featureImpacts?.slice(0, 3),
          }
        : undefined;

      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          message: trimmed,
          history,
          score_context: scoreContext,
        }),
      });

      const data = await res.json();
      if (data.reply) {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: data.reply },
        ]);
      } else {
        throw new Error(data.error || "No reply");
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Sorry, I'm having trouble connecting right now. Please try again in a moment.",
        },
      ]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8 pb-16 flex flex-col h-[calc(100vh-4rem)]">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6 border-b border-border pb-4">
        <div className="h-10 w-10 rounded-full bg-accent/10 border border-accent/30 flex items-center justify-center">
          <BrainCircuit size={20} className="text-accent" />
        </div>
        <div>
          <h1 className="text-xl font-serif font-bold text-text">AI Financial Coach</h1>
          <p className="text-xs text-muted">
            Powered by Llama 3 · India-specific credit advice
          </p>
        </div>
        {result && (
          <div className="ml-auto text-right">
            <div className="text-xs text-muted">Your score</div>
            <div className="text-lg font-bold text-accent">{result.score}</div>
          </div>
        )}
      </div>

      {/* Starter prompts — only shown when only the greeting exists */}
      {messages.length === 1 && (
        <div className="mb-4 flex flex-wrap gap-2">
          {STARTER_PROMPTS.map((p) => (
            <button
              key={p}
              onClick={() => sendMessage(p)}
              className="text-xs px-3 py-1.5 rounded-full border border-border text-muted hover:border-accent/40 hover:text-accent transition-colors"
            >
              {p}
            </button>
          ))}
        </div>
      )}

      {/* Message thread */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-1">
        <AnimatePresence initial={false}>
          {messages.map((msg, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
            >
              {/* Avatar */}
              <div
                className={`h-8 w-8 shrink-0 rounded-full flex items-center justify-center border ${
                  msg.role === "assistant"
                    ? "bg-accent/10 border-accent/30"
                    : "bg-card border-border"
                }`}
              >
                {msg.role === "assistant" ? (
                  <Sparkles size={14} className="text-accent" />
                ) : (
                  <User size={14} className="text-muted" />
                )}
              </div>

              {/* Bubble */}
              <div
                className={`max-w-[80%] px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                  msg.role === "assistant"
                    ? "bg-card border border-border text-text rounded-tl-sm"
                    : "bg-accent/10 border border-accent/20 text-text rounded-tr-sm"
                }`}
              >
                {msg.content}
              </div>
            </motion.div>
          ))}

          {/* Typing indicator */}
          {isLoading && (
            <motion.div
              key="typing"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex gap-3"
            >
              <div className="h-8 w-8 shrink-0 rounded-full bg-accent/10 border border-accent/30 flex items-center justify-center">
                <Sparkles size={14} className="text-accent" />
              </div>
              <div className="px-4 py-3 rounded-2xl rounded-tl-sm bg-card border border-border">
                <Loader2 size={16} className="text-muted animate-spin" />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="mt-4 flex gap-3 items-end">
        <Card className="flex-1 p-0 overflow-hidden">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about your score, EMI, loan eligibility..."
            rows={1}
            className="w-full bg-transparent px-4 py-3 text-sm text-text placeholder:text-muted resize-none focus:outline-none"
            style={{ minHeight: "48px", maxHeight: "120px" }}
            onInput={(e) => {
              const el = e.currentTarget;
              el.style.height = "auto";
              el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
            }}
          />
        </Card>
        <Button
          variant="accent"
          size="icon"
          onClick={() => sendMessage(input)}
          disabled={isLoading || !input.trim()}
          className="h-12 w-12 shrink-0"
        >
          <Send size={16} />
        </Button>
      </div>

      <p className="text-center text-[10px] text-muted mt-2">
        Press Enter to send · Shift+Enter for new line · Advice is for educational purposes only
      </p>
    </div>
  );
}

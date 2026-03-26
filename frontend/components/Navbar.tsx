"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "next-themes";
import { useState, useEffect } from "react";
import { Button } from "./ui/Button";
import { Sun, Moon, Activity } from "lucide-react";
import { motion } from "framer-motion";

export function Navbar() {
  const pathname = usePathname();
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const NAV_LINKS = [
    { name: "Dashboard", path: "/home"    },
    { name: "History",   path: "/history" },
    { name: "Chat",      path: "/chat"    },
    { name: "Loans",     path: "/loans"   },
    { name: "API Docs",  path: "/api"     },
  ];

  return (
    <nav className="sticky top-0 z-50 w-full border-b border-border bg-bg/60 backdrop-blur-2xl transition-colors duration-300">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-8">
          <Link href="/" className="flex items-center gap-2 group">
            <motion.div
              whileHover={{ scale: 1.05, rotate: 90 }}
              transition={{ type: "spring", stiffness: 300, damping: 20 }}
              className="h-6 w-6 rounded-sm bg-accent/10 flex items-center justify-center text-accent ring-1 ring-accent/30 shadow-[0_0_15px_rgba(5,255,170,0.15)]"
            >
              <Activity size={14} className="font-bold" />
            </motion.div>
            <span className="text-xl font-serif font-bold tracking-widest uppercase text-text group-hover:text-accent transition-colors">
              NeoScore
            </span>
          </Link>

          <div className="hidden md:flex ml-8 space-x-6 items-center">
            {NAV_LINKS.map((link) => (
              <Link
                key={link.path}
                href={link.path}
                className={`text-xs font-semibold tracking-widest uppercase transition-all duration-300 relative ${
                  pathname.startsWith(link.path)
                    ? "text-accent after:absolute after:bottom-[-4px] after:left-0 after:w-full after:h-[1px] after:bg-accent"
                    : "text-muted hover:text-text"
                }`}
              >
                {link.name}
              </Link>
            ))}
          </div>
        </div>

        <div className="flex items-center space-x-6">
          {mounted ? (
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              className="p-1.5 rounded-sm bg-transparent border border-border text-muted hover:text-accent hover:border-accent/40 transition-all"
              aria-label="Toggle theme"
            >
              {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
            </motion.button>
          ) : (
            <div className="w-[30px] h-[30px] rounded-sm bg-card border border-border opacity-50 block" />
          )}
          <Button
            variant="outline"
            size="sm"
            className="hidden sm:inline-flex rounded-sm"
            onClick={() => { window.location.href = "http://localhost:5000/auth/login/google"; }}
          >
            Sign In
          </Button>
          <Button variant="accent" size="sm" className="hidden sm:inline-flex group rounded-sm tracking-widest">
            Get Pro <Activity size={14} className="ml-1.5 opacity-70 group-hover:animate-pulse" />
          </Button>
        </div>
      </div>
    </nav>
  );
}

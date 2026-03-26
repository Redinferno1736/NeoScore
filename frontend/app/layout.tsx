import type { Metadata } from "next";
import { Inter, Playfair_Display } from "next/font/google";
import "./globals.css";
import { Navbar } from "@/components/Navbar";
import { ThemeProvider } from "@/components/ThemeProvider";
import { AnimatedBackground } from "@/components/AnimatedBackground";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const playfair = Playfair_Display({
  variable: "--font-playfair",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "NeoScore | Explainable AI Credit Dashboard",
  description: "Understand your credit. Improve it instantly with AI.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${playfair.variable} antialiased`}
      suppressHydrationWarning
    >
      <body className="flex flex-col bg-bg text-text selection:bg-accent/30 selection:text-white transition-colors duration-300">
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange={false}
        >
          <AnimatedBackground />
          <Navbar />
          <main className="flex-1 w-full mx-auto">
            {children}
          </main>
        </ThemeProvider>
      </body>
    </html>
  );
}

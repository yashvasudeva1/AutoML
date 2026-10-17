import React, { useState } from "react";
import { 
  Lock, Mail, User, ShieldCheck, ArrowRight, Sparkles, Database,
  TrendingUp, Play, CheckCircle, HelpCircle, Eye, EyeOff, AlertCircle, 
  Loader2, ChevronDown
} from "lucide-react";
import DistributionBackground from "./DistributionBackground";
import { signInWithGoogle } from "../utils/firebaseAuth";

interface LandingPageProps {
  onLoginSuccess: (email: string, role?: string) => void;
}

export default function LandingPage({ onLoginSuccess }: LandingPageProps) {
  // Auth navigation states: "landing" | "login" | "signup"
  const [currentView, setCurrentView] = useState<"landing" | "login" | "signup">("landing");

  // Form Inputs
  const [loginEmail, setLoginEmail] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [signUpEmail, setSignUpEmail] = useState("");
  const [signUpName, setSignUpName] = useState("");
  const [signUpPassword, setSignUpPassword] = useState("");
  const [signUpPasswordConfirm, setSignUpPasswordConfirm] = useState("");

  // UI Utilities
  const [showPassword, setShowPassword] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);

  // Read saved users or initialize empty store
  const getSavedUsers = (): Array<{ email: string; name: string; pass: string }> => {
    try {
      const stored = localStorage.getItem("automl_user_accounts");
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  };

  // Google OAuth Sign-In handler
  const handleGoogleSignIn = async () => {
    setErrorMsg(null);
    setIsGoogleLoading(true);
    try {
      const result = await signInWithGoogle();
      if (result.user && result.user.email) {
        // Sync or register user in persistent DB
        try {
          await fetch("/api/auth/register", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              email: result.user.email,
              name: result.user.displayName || result.user.email.split("@")[0],
              password: `oauth_${result.user.uid}`
            })
          });
        } catch {}

        onLoginSuccess(
          result.user.email,
          result.user.email.toLowerCase() === "vasudevyash@gmail.com" ? "admin" : "user"
        );
      }
    } catch (err: any) {
      console.error("Google OAuth error:", err);
      if (
        err.code === "auth/invalid-api-key" || 
        err.code === "auth/api-key-not-valid" || 
        err.message?.includes("API key") || 
        err.message?.includes("not configured")
      ) {
        setErrorMsg(
          "Google Sign-In requires your Firebase Web API Key. Please add your Firebase project settings to Render environment variables (VITE_FIREBASE_API_KEY) or sign in with your email/password."
        );
      } else if (err.code === "auth/unauthorized-domain") {
        setErrorMsg(
          "This domain is not authorized in Firebase. Please add this domain to Firebase Console > Authentication > Settings > Authorized Domains."
        );
      } else if (err.code === "auth/popup-blocked") {
        setErrorMsg("Google Sign-In popup was blocked by browser settings. Please allow popups for this site.");
      } else if (err.code === "auth/popup-closed-by-user") {
        setErrorMsg("Sign-in was cancelled. Please try again.");
      } else {
        setErrorMsg(err.message || "Google authentication failed. Please try again or sign in with email/password.");
      }
    } finally {
      setIsGoogleLoading(false);
    }
  };


  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg(null);
    setSuccessMsg(null);

    if (!signUpEmail.trim() || !signUpName.trim() || !signUpPassword) {
      setErrorMsg("All fields are required.");
      return;
    }

    if (signUpPassword.length < 6) {
      setErrorMsg("Password must be at least 6 characters long.");
      return;
    }

    if (signUpPassword !== signUpPasswordConfirm) {
      setErrorMsg("Passwords do not match.");
      return;
    }

    try {
      const res = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: signUpEmail.trim(),
          name: signUpName.trim(),
          password: signUpPassword
        })
      });
      const data = await res.json();
      if (!res.ok) {
        setErrorMsg(data.error || "Failed to register account.");
        return;
      }

      // Also mirror to client localStorage
      const saved = getSavedUsers();
      if (!saved.some(u => u.email.toLowerCase() === signUpEmail.toLowerCase())) {
        saved.push({
          email: signUpEmail,
          name: signUpName,
          pass: signUpPassword
        });
        localStorage.setItem("automl_user_accounts", JSON.stringify(saved));
      }

      setSuccessMsg("Account successfully registered! You can now log in.");
      setTimeout(() => {
        setLoginEmail(signUpEmail);
        setCurrentView("login");
        setSuccessMsg(null);
      }, 1000);
    } catch {
      // Offline fallback: save to localStorage
      const saved = getSavedUsers();
      if (saved.some(u => u.email.toLowerCase() === signUpEmail.toLowerCase())) {
        setErrorMsg("An account with this email address already exists.");
        return;
      }
      saved.push({
        email: signUpEmail,
        name: signUpName,
        pass: signUpPassword
      });
      localStorage.setItem("automl_user_accounts", JSON.stringify(saved));
      setSuccessMsg("Account registered locally. You can now log in.");
      setTimeout(() => {
        setLoginEmail(signUpEmail);
        setCurrentView("login");
        setSuccessMsg(null);
      }, 1000);
    }
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg(null);

    if (!loginEmail.trim() || !loginPassword) {
      setErrorMsg("Please enter both email and password.");
      return;
    }

    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: loginEmail.trim(),
          password: loginPassword
        })
      });
      const data = await res.json();
      if (res.ok && data.success && data.user) {
        onLoginSuccess(data.user.email, data.user.role || "user");
        return;
      }
      if (!res.ok) {
        // Check local storage fallback before reporting error
        const saved = getSavedUsers();
        const match = saved.find(
          u => u.email.toLowerCase() === loginEmail.toLowerCase() && u.pass === loginPassword
        );
        if (match) {
          onLoginSuccess(match.email, match.email.toLowerCase() === "vasudevyash@gmail.com" ? "admin" : "user");
          return;
        }
        setErrorMsg(data.error || "Invalid email address or password combination.");
        return;
      }
    } catch {
      // Offline fallback check
      const saved = getSavedUsers();
      const match = saved.find(
        u => u.email.toLowerCase() === loginEmail.toLowerCase() && u.pass === loginPassword
      );
      if (match) {
        onLoginSuccess(match.email, match.email.toLowerCase() === "vasudevyash@gmail.com" ? "admin" : "user");
        return;
      }
      setErrorMsg("Invalid email address or password combination.");
    }
  };

  const handleScrollToSandbox = () => {
    const el = document.getElementById("sandbox-section");
    if (el) {
      el.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="relative min-h-screen bg-[#FAF9F5] text-[#1F1E1D] overflow-x-hidden font-sans">
      
      {/* Subtle Static Background Grid */}
      <div 
        className="absolute inset-0 pointer-events-none opacity-40"
        style={{
          backgroundImage: "radial-gradient(#E5E2D9 1px, transparent 1px)",
          backgroundSize: "24px 24px"
        }}
      />

      {/* Warm Subtle Header */}
      <header className="border-b border-[#E5E2D9] bg-[#FAF9F5]/90 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-[#0284C7]/10 flex items-center justify-center text-[#0284C7]">
              <Database className="w-4 h-4" />
            </div>
            <div>
              <span className="font-serif text-base font-semibold tracking-tight text-[#1F1E1D]">
                AutoML Studio
              </span>
              <span className="text-[10px] text-[#73726C] block -mt-1">
                Data Science Workbench
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-2.5">
            {currentView === "landing" ? (
              <>
                <button
                  onClick={() => { setErrorMsg(null); setCurrentView("login"); }}
                  className="px-3.5 py-1.5 text-xs font-medium text-[#73726C] hover:text-[#1F1E1D] rounded-lg transition-colors cursor-pointer"
                >
                  Sign In
                </button>
                <button
                  onClick={() => { setErrorMsg(null); setCurrentView("signup"); }}
                  className="px-4 py-1.5 text-xs font-medium bg-[#0284C7] hover:bg-[#0369A1] text-white rounded-lg shadow-xs transition-colors cursor-pointer"
                >
                  Get Started
                </button>
              </>
            ) : (
              <button
                onClick={() => { setErrorMsg(null); setCurrentView("landing"); }}
                className="px-3.5 py-1.5 text-xs font-medium text-[#73726C] hover:text-[#1F1E1D] rounded-lg hover:bg-[#F0EEE6] transition-colors cursor-pointer"
              >
                Back to Home
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Hero / Landing View */}
      {currentView === "landing" && (
        <div className="relative z-10">
          
          {/* ========================================================================= */}
          {/* SECTION 1: HERO VIEWPORT (Dedicated full above-the-fold screen) */}
          {/* ========================================================================= */}
          <section className="min-h-[calc(100vh-4rem)] flex flex-col justify-center items-center px-4 sm:px-6 lg:px-8 max-w-5xl mx-auto py-12 relative text-center">
            
            <div className="space-y-6 max-w-3xl mx-auto my-auto">
              
              {/* Product Badge */}
              <div className="inline-flex items-center gap-2 px-3.5 py-1.5 bg-[#F0F9FF] border border-[#BAE6FD] rounded-full text-xs text-[#0284C7] font-medium shadow-2xs">
                <Sparkles className="h-3.5 w-3.5" />
                <span>Automated EDA & Polars Vector Engine</span>
              </div>

              {/* Main Headline */}
              <h1 className="font-serif text-4xl sm:text-5xl lg:text-6xl font-normal tracking-tight text-[#1F1E1D] leading-[1.15]">
                Statistical rigour meets <span className="italic font-serif text-[#0284C7]">high-throughput</span> AutoML.
              </h1>

              {/* Subtitle Description */}
              <p className="text-base sm:text-lg text-[#73726C] leading-relaxed max-w-2xl mx-auto">
                Ingest tabular datasets, compute bivariate correlations and hypothesis tests, clean out-of-core pipelines with Polars, and benchmark machine learning models.
              </p>

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-3">
                <button
                  onClick={() => { setErrorMsg(null); setCurrentView("signup"); }}
                  className="w-full sm:w-auto px-6 py-3 bg-[#0284C7] hover:bg-[#0369A1] text-white text-sm font-medium rounded-xl shadow-xs transition-all cursor-pointer flex items-center justify-center space-x-2 hover:scale-[1.01]"
                >
                  <span>Create Free Workspace</span>
                  <ArrowRight className="h-4 w-4" />
                </button>

                <button
                  onClick={handleGoogleSignIn}
                  disabled={isGoogleLoading}
                  className="w-full sm:w-auto px-5 py-2.5 bg-white hover:bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] text-xs font-medium rounded-xl shadow-xs transition-colors cursor-pointer flex items-center justify-center space-x-2.5"
                >
                  <svg className="w-4 h-4" viewBox="0 0 24 24">
                    <path fill="#4285F4" d="M23.745 12.27c0-.7-.06-1.4-.19-2.07H12v4.51h6.6c-.29 1.52-1.14 2.82-2.4 3.68v3.05h3.88c2.27-2.09 3.665-5.17 3.665-9.17z"/>
                    <path fill="#34A853" d="M12 24c3.24 0 5.95-1.08 7.93-2.91l-3.88-3.05c-1.08.72-2.45 1.16-4.05 1.16-3.12 0-5.77-2.1-6.72-4.93H1.25v3.15C3.26 21.36 7.33 24 12 24z"/>
                    <path fill="#FBBC05" d="M5.28 14.27c-.25-.72-.38-1.49-.38-2.27s.13-1.55.38-2.27V6.58H1.25C.45 8.18 0 10.03 0 12s.45 3.82 1.25 5.42l4.03-3.15z"/>
                    <path fill="#EA4335" d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.42-3.42C17.95 1.19 15.24 0 12 0 7.33 0 3.26 2.64 1.25 6.58l4.03 3.15c.95-2.83 3.6-4.98 6.72-4.98z"/>
                  </svg>
                  <span>Continue with Google</span>
                </button>

                <button
                  onClick={() => { setErrorMsg(null); setCurrentView("login"); }}
                  className="w-full sm:w-auto px-5 py-2.5 bg-white hover:bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] text-xs font-medium rounded-xl shadow-xs transition-colors cursor-pointer"
                >
                  Sign In
                </button>
              </div>
            </div>

            {/* Scroll indicator prompting the user to scroll to section 2 */}
            <div className="pt-10 pb-4">
              <button
                onClick={handleScrollToSandbox}
                className="inline-flex flex-col items-center justify-center text-[#73726C] hover:text-[#0284C7] transition-colors cursor-pointer group"
              >
                <span className="text-[10px] font-mono tracking-wider uppercase text-[#9E9D96] group-hover:text-[#0284C7]">
                  Scroll to Interactive Sandbox
                </span>
                <ChevronDown className="h-4 w-4 mt-1 text-[#0284C7] group-hover:translate-y-0.5 transition-transform" />
              </button>
            </div>
          </section>

          {/* ========================================================================= */}
          {/* SECTION 2: LIVE MATHEMATICAL SANDBOX (Spaciously below the fold) */}
          {/* ========================================================================= */}
          <section id="sandbox-section" className="pt-20 sm:pt-28 pb-20 px-4 sm:px-6 lg:px-8 max-w-6xl mx-auto border-t border-[#E5E2D9]/70 space-y-8">
            <div className="text-center space-y-1.5 max-w-xl mx-auto">
              <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#FAF9F5] border border-[#E5E2D9] text-[10px] font-mono uppercase tracking-wider text-[#0284C7] font-semibold">
                <Sparkles className="h-3 w-3" />
                <span>Real-Time Math Engine</span>
              </div>
              <h2 className="font-serif text-2xl sm:text-3xl font-normal text-[#1F1E1D]">
                Live Mathematical Sandbox
              </h2>
              <p className="text-xs sm:text-sm text-[#73726C] leading-relaxed">
                Explore continuous probability density functions, adjust mean (μ) and variance (σ), and inspect moment profiles in real time.
              </p>
            </div>

            <div className="shadow-xs rounded-2xl bg-white border border-[#E5E2D9] p-4 sm:p-6">
              <DistributionBackground />
            </div>
          </section>

          {/* ========================================================================= */}
          {/* SECTION 3: CAPABILITIES & ARCHITECTURAL PILLARS */}
          {/* ========================================================================= */}
          <section className="pt-8 pb-28 px-4 sm:px-6 lg:px-8 max-w-6xl mx-auto border-t border-[#E5E2D9]/70 space-y-8">
            <div className="text-center space-y-1.5 max-w-xl mx-auto">
              <span className="text-[10px] font-mono uppercase tracking-wider text-[#73726C] font-semibold">
                Core Capabilities
              </span>
              <h2 className="font-serif text-2xl sm:text-3xl font-normal text-[#1F1E1D]">
                Engineered for Rigorous Data Workloads
              </h2>
            </div>

            <div className="grid sm:grid-cols-3 gap-6">
              <div className="bg-white border border-[#E5E2D9] rounded-2xl p-6 space-y-3 shadow-xs hover:border-[#0284C7]/50 transition-colors">
                <div className="w-10 h-10 rounded-xl bg-[#0284C7]/10 flex items-center justify-center text-[#0284C7]">
                  <TrendingUp className="h-5 w-5" />
                </div>
                <h3 className="font-serif text-base font-medium text-[#1F1E1D]">Exploratory EDA</h3>
                <p className="text-xs text-[#73726C] leading-relaxed">
                  Automated schema profiling, Pearson & Spearman correlation heatmaps, histogram distributions, and bivariate trendlines.
                </p>
              </div>

              <div className="bg-white border border-[#E5E2D9] rounded-2xl p-6 space-y-3 shadow-xs hover:border-[#0284C7]/50 transition-colors">
                <div className="w-10 h-10 rounded-xl bg-[#0284C7]/10 flex items-center justify-center text-[#0284C7]">
                  <Database className="h-5 w-5" />
                </div>
                <h3 className="font-serif text-base font-medium text-[#1F1E1D]">Polars Vector Engine</h3>
                <p className="text-xs text-[#73726C] leading-relaxed">
                  Execute streaming out-of-core pipeline cleaning, IQR outlier bounds, null imputation, and one-click Python script export.
                </p>
              </div>

              <div className="bg-white border border-[#E5E2D9] rounded-2xl p-6 space-y-3 shadow-xs hover:border-[#0284C7]/50 transition-colors">
                <div className="w-10 h-10 rounded-xl bg-[#0284C7]/10 flex items-center justify-center text-[#0284C7]">
                  <Sparkles className="h-5 w-5" />
                </div>
                <h3 className="font-serif text-base font-medium text-[#1F1E1D]">AutoML Tournament</h3>
                <p className="text-xs text-[#73726C] leading-relaxed">
                  Benchmark Random Forests, Gradient Boosters, and Logistic Regressions with K-Fold cross-validation and feature importance.
                </p>
              </div>
            </div>

            {/* Bottom CTA Card */}
            <div className="bg-gradient-to-r from-white via-white to-[#F0F9FF] border border-[#E5E2D9] rounded-2xl p-6 sm:p-8 shadow-xs flex flex-col sm:flex-row items-center justify-between gap-4 text-center sm:text-left mt-8">
              <div className="space-y-1">
                <h3 className="font-serif text-lg font-medium text-[#1F1E1D]">
                  Ready to profile and model your tabular data?
                </h3>
                <p className="text-xs text-[#73726C]">
                  Sign in to launch your high-throughput analytics workbench with Gemini & Hugging Face.
                </p>
              </div>
              <div className="flex flex-wrap items-center justify-center gap-2.5 shrink-0">
                <button
                  onClick={() => { setErrorMsg(null); setCurrentView("login"); }}
                  className="px-4 py-2 bg-white hover:bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] text-xs font-medium rounded-xl shadow-2xs transition-colors cursor-pointer"
                >
                  Sign In
                </button>
                <button
                  onClick={() => { setErrorMsg(null); setCurrentView("signup"); }}
                  className="px-4 py-2 bg-[#0284C7] hover:bg-[#0369A1] text-white text-xs font-medium rounded-xl shadow-xs transition-colors cursor-pointer"
                >
                  Create Free Account
                </button>
              </div>
            </div>
          </section>
        </div>
      )}

      {/* Login & Sign-up Modal Container */}
      {currentView !== "landing" && (
        <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center px-4 py-12 relative z-10">
          <div className="w-full max-w-md bg-white border border-[#E5E2D9] rounded-2xl shadow-lg p-6 sm:p-8 space-y-6">
            
            {/* Header */}
            <div className="text-center space-y-1.5">
              <div className="w-10 h-10 rounded-xl bg-[#0284C7]/10 flex items-center justify-center text-[#0284C7] mx-auto mb-2">
                <Lock className="h-5 w-5" />
              </div>
              <h2 className="font-serif text-2xl font-normal text-[#1F1E1D]">
                {currentView === "login" ? "Sign in to Workbench" : "Create your account"}
              </h2>
              <p className="text-xs text-[#73726C]">
                {currentView === "login" 
                  ? "Access your analytical pipelines, models, and datasets" 
                  : "Start analyzing data with automated machine learning"}
              </p>
            </div>

            {/* Official Google OAuth Sign In Button */}
            <div className="space-y-3">
              <button
                type="button"
                onClick={handleGoogleSignIn}
                disabled={isGoogleLoading}
                className="w-full py-2.5 px-4 bg-white hover:bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] text-xs font-medium rounded-xl shadow-xs transition-colors cursor-pointer flex items-center justify-center space-x-3 disabled:opacity-50"
              >
                {isGoogleLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin text-[#0284C7]" />
                ) : (
                  <svg className="w-4 h-4 shrink-0" viewBox="0 0 24 24">
                    <path fill="#4285F4" d="M23.745 12.27c0-.7-.06-1.4-.19-2.07H12v4.51h6.6c-.29 1.52-1.14 2.82-2.4 3.68v3.05h3.88c2.27-2.09 3.665-5.17 3.665-9.17z"/>
                    <path fill="#34A853" d="M12 24c3.24 0 5.95-1.08 7.93-2.91l-3.88-3.05c-1.08.72-2.45 1.16-4.05 1.16-3.12 0-5.77-2.1-6.72-4.93H1.25v3.15C3.26 21.36 7.33 24 12 24z"/>
                    <path fill="#FBBC05" d="M5.28 14.27c-.25-.72-.38-1.49-.38-2.27s.13-1.55.38-2.27V6.58H1.25C.45 8.18 0 10.03 0 12s.45 3.82 1.25 5.42l4.03-3.15z"/>
                    <path fill="#EA4335" d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.42-3.42C17.95 1.19 15.24 0 12 0 7.33 0 3.26 2.64 1.25 6.58l4.03 3.15c.95-2.83 3.6-4.98 6.72-4.98z"/>
                  </svg>
                )}
                <span>Continue with Google</span>
              </button>

              <div className="relative flex items-center justify-center">
                <div className="border-t border-[#E5E2D9] w-full" />
                <span className="bg-white px-2.5 text-[11px] text-[#73726C] uppercase font-mono relative">
                  or email & password
                </span>
              </div>
            </div>

            {/* Error / Success Notifications */}
            {errorMsg && (
              <div className="p-3 bg-rose-50 border border-rose-200 text-rose-700 text-xs rounded-xl flex items-center space-x-2">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span>{errorMsg}</span>
              </div>
            )}

            {successMsg && (
              <div className="p-3 bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs rounded-xl flex items-center space-x-2">
                <CheckCircle className="h-4 w-4 shrink-0" />
                <span>{successMsg}</span>
              </div>
            )}

            {/* Form View 1: Login */}
            {currentView === "login" && (
              <form onSubmit={handleLogin} className="space-y-4">


                <div className="space-y-1">
                  <label className="text-xs font-medium text-[#1F1E1D]">Email Address</label>
                  <div className="relative">
                    <Mail className="h-4 w-4 absolute left-3 top-3 text-[#73726C]" />
                    <input
                      type="email"
                      required
                      value={loginEmail}
                      onChange={(e) => setLoginEmail(e.target.value)}
                      placeholder="analyst@domain.com"
                      className="w-full bg-[#FAF9F5] border border-[#E5E2D9] rounded-xl pl-9 pr-3 py-2 text-xs text-[#1F1E1D] focus:border-[#0284C7] focus:ring-1 focus:ring-[#0284C7] outline-none transition-all"
                    />
                  </div>
                </div>

                <div className="space-y-1">
                  <label className="text-xs font-medium text-[#1F1E1D]">Password</label>
                  <div className="relative">
                    <Lock className="h-4 w-4 absolute left-3 top-3 text-[#73726C]" />
                    <input
                      type={showPassword ? "text" : "password"}
                      required
                      value={loginPassword}
                      onChange={(e) => setLoginPassword(e.target.value)}
                      placeholder="••••••••"
                      className="w-full bg-[#FAF9F5] border border-[#E5E2D9] rounded-xl pl-9 pr-9 py-2 text-xs text-[#1F1E1D] focus:border-[#0284C7] focus:ring-1 focus:ring-[#0284C7] outline-none transition-all"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-3 text-[#73726C] hover:text-[#1F1E1D]"
                    >
                      {showPassword ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                    </button>
                  </div>
                </div>

                <button
                  type="submit"
                  className="w-full py-2.5 bg-[#0284C7] hover:bg-[#0369A1] text-white text-xs font-medium rounded-xl transition-colors cursor-pointer shadow-xs font-sans"
                >
                  Sign In to Studio
                </button>

                <div className="pt-2 text-center text-xs text-[#73726C]">
                  Don't have an account?{" "}
                  <button
                    type="button"
                    onClick={() => { setErrorMsg(null); setCurrentView("signup"); }}
                    className="text-[#0284C7] hover:underline font-medium cursor-pointer"
                  >
                    Sign up
                  </button>
                </div>
              </form>
            )}

            {/* Form View 2: Sign-up */}
            {currentView === "signup" && (
              <form onSubmit={handleSignUp} className="space-y-4">
                <div className="space-y-1">
                  <label className="text-xs font-medium text-[#1F1E1D]">Full Name</label>
                  <div className="relative">
                    <User className="h-4 w-4 absolute left-3 top-3 text-[#73726C]" />
                    <input
                      type="text"
                      required
                      value={signUpName}
                      onChange={(e) => setSignUpName(e.target.value)}
                      placeholder="Jane Doe"
                      className="w-full bg-[#FAF9F5] border border-[#E5E2D9] rounded-xl pl-9 pr-3 py-2 text-xs text-[#1F1E1D] focus:border-[#0284C7] focus:ring-1 focus:ring-[#0284C7] outline-none"
                    />
                  </div>
                </div>

                <div className="space-y-1">
                  <label className="text-xs font-medium text-[#1F1E1D]">Email Address</label>
                  <div className="relative">
                    <Mail className="h-4 w-4 absolute left-3 top-3 text-[#73726C]" />
                    <input
                      type="email"
                      required
                      value={signUpEmail}
                      onChange={(e) => setSignUpEmail(e.target.value)}
                      placeholder="analyst@domain.com"
                      className="w-full bg-[#FAF9F5] border border-[#E5E2D9] rounded-xl pl-9 pr-3 py-2 text-xs text-[#1F1E1D] focus:border-[#0284C7] focus:ring-1 focus:ring-[#0284C7] outline-none"
                    />
                  </div>
                </div>

                <div className="space-y-1">
                  <label className="text-xs font-medium text-[#1F1E1D]">Create Password</label>
                  <div className="relative">
                    <Lock className="h-4 w-4 absolute left-3 top-3 text-[#73726C]" />
                    <input
                      type={showPassword ? "text" : "password"}
                      required
                      value={signUpPassword}
                      onChange={(e) => setSignUpPassword(e.target.value)}
                      placeholder="At least 6 characters"
                      className="w-full bg-[#FAF9F5] border border-[#E5E2D9] rounded-xl pl-9 pr-3 py-2 text-xs text-[#1F1E1D] focus:border-[#0284C7] focus:ring-1 focus:ring-[#0284C7] outline-none"
                    />
                  </div>
                </div>

                <div className="space-y-1">
                  <label className="text-xs font-medium text-[#1F1E1D]">Confirm Password</label>
                  <div className="relative">
                    <Lock className="h-4 w-4 absolute left-3 top-3 text-[#73726C]" />
                    <input
                      type={showPassword ? "text" : "password"}
                      required
                      value={signUpPasswordConfirm}
                      onChange={(e) => setSignUpPasswordConfirm(e.target.value)}
                      placeholder="Repeat password"
                      className="w-full bg-[#FAF9F5] border border-[#E5E2D9] rounded-xl pl-9 pr-3 py-2 text-xs text-[#1F1E1D] focus:border-[#0284C7] focus:ring-1 focus:ring-[#0284C7] outline-none"
                    />
                  </div>
                </div>

                <button
                  type="submit"
                  className="w-full py-2.5 bg-[#0284C7] hover:bg-[#0369A1] text-white text-xs font-medium rounded-xl transition-colors cursor-pointer shadow-xs"
                >
                  Create Account
                </button>

                <div className="pt-2 text-center text-xs text-[#73726C]">
                  Already have an account?{" "}
                  <button
                    type="button"
                    onClick={() => { setErrorMsg(null); setCurrentView("login"); }}
                    className="text-[#0284C7] hover:underline font-medium cursor-pointer"
                  >
                    Sign in
                  </button>
                </div>
              </form>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

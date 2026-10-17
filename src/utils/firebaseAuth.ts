import { initializeApp, getApps, getApp, deleteApp, FirebaseApp } from "firebase/app";
import { 
  getAuth, 
  signInWithPopup, 
  GoogleAuthProvider, 
  onAuthStateChanged, 
  signOut as firebaseSignOut,
  User,
  Auth
} from "firebase/auth";
import firebaseConfig from "../../firebase-applet-config.json";

export interface FirebaseAppConfig {
  apiKey: string;
  authDomain: string;
  projectId: string;
  storageBucket?: string;
  messagingSenderId?: string;
  appId?: string;
}

export function isConfigValid(cfg?: Partial<FirebaseAppConfig> | null): boolean {
  if (!cfg) return false;
  const key = cfg.apiKey;
  const pid = cfg.projectId;
  return Boolean(
    key && typeof key === "string" && key.trim() !== "" && !key.includes("your-") &&
    pid && typeof pid === "string" && pid.trim() !== "" && !pid.includes("your-")
  );
}

// Build-time environment or config file fallback
const staticConfig: FirebaseAppConfig = {
  apiKey: (typeof import.meta !== "undefined" && import.meta.env?.VITE_FIREBASE_API_KEY) || firebaseConfig.apiKey || "",
  authDomain: (typeof import.meta !== "undefined" && import.meta.env?.VITE_FIREBASE_AUTH_DOMAIN) || firebaseConfig.authDomain || "",
  projectId: (typeof import.meta !== "undefined" && import.meta.env?.VITE_FIREBASE_PROJECT_ID) || firebaseConfig.projectId || "",
  storageBucket: (typeof import.meta !== "undefined" && import.meta.env?.VITE_FIREBASE_STORAGE_BUCKET) || firebaseConfig.storageBucket || "",
  messagingSenderId: (typeof import.meta !== "undefined" && import.meta.env?.VITE_FIREBASE_MESSAGING_SENDER_ID) || firebaseConfig.messagingSenderId || "",
  appId: (typeof import.meta !== "undefined" && import.meta.env?.VITE_FIREBASE_APP_ID) || firebaseConfig.appId || "",
};

let activeApp: FirebaseApp | null = null;
let activeAuth: Auth | null = null;

export async function getOrInitAuth(): Promise<{ auth: Auth | null; isConfigured: boolean }> {
  if (activeAuth && isConfigValid(activeApp?.options as any)) {
    return { auth: activeAuth, isConfigured: true };
  }

  // 1. Check if static config is valid
  if (isConfigValid(staticConfig)) {
    try {
      activeApp = getApps().length > 0 ? getApp() : initializeApp(staticConfig);
      activeAuth = getAuth(activeApp);
      return { auth: activeAuth, isConfigured: true };
    } catch (err) {
      console.warn("Failed initializing static Firebase app:", err);
    }
  }

  // 2. Try fetching server runtime config
  try {
    const res = await fetch("/api/auth/firebase-config");
    if (res.ok) {
      const serverCfg = await res.json();
      if (serverCfg.isConfigured && isConfigValid(serverCfg)) {
        if (getApps().length > 0) {
          try {
            await deleteApp(getApp());
          } catch {}
        }
        activeApp = initializeApp(serverCfg);
        activeAuth = getAuth(activeApp);
        return { auth: activeAuth, isConfigured: true };
      }
    }
  } catch {}

  return { auth: null, isConfigured: false };
}

const googleProvider = new GoogleAuthProvider();
googleProvider.addScope("openid");
googleProvider.addScope("https://www.googleapis.com/auth/userinfo.email");
googleProvider.addScope("https://www.googleapis.com/auth/userinfo.profile");

let cachedAccessToken: string | null = null;
let isSigningIn = false;

export async function signInWithGoogle(): Promise<{ user: User; token: string | null }> {
  const { auth, isConfigured } = await getOrInitAuth();
  if (!isConfigured || !auth) {
    throw new Error(
      "Firebase Authentication is not configured yet. Please provide your Firebase Web API Key and Project ID to enable Google Sign-In."
    );
  }

  isSigningIn = true;
  try {
    const result = await signInWithPopup(auth, googleProvider);
    const credential = GoogleAuthProvider.credentialFromResult(result);
    cachedAccessToken = credential?.accessToken || null;
    return { user: result.user, token: cachedAccessToken };
  } finally {
    isSigningIn = false;
  }
}

export async function signOutGoogle(): Promise<void> {
  cachedAccessToken = null;
  if (activeAuth) {
    try {
      await firebaseSignOut(activeAuth);
    } catch {}
  }
}

export function getCachedToken(): string | null {
  return cachedAccessToken;
}

export function subscribeToAuth(
  onUserChanged: (user: User | null, token: string | null) => void
) {
  if (activeAuth) {
    return onAuthStateChanged(activeAuth, (user) => {
      if (!user) {
        cachedAccessToken = null;
        onUserChanged(null, null);
      } else {
        onUserChanged(user, cachedAccessToken);
      }
    });
  }

  getOrInitAuth().then(({ auth }) => {
    if (auth) {
      onAuthStateChanged(auth, (user) => {
        if (!user) {
          cachedAccessToken = null;
          onUserChanged(null, null);
        } else {
          onUserChanged(user, cachedAccessToken);
        }
      });
    } else {
      onUserChanged(null, null);
    }
  });

  return () => {};
}

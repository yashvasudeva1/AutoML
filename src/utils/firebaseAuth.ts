import { initializeApp, getApps, getApp } from "firebase/app";
import { 
  getAuth, 
  signInWithPopup, 
  GoogleAuthProvider, 
  onAuthStateChanged, 
  signOut as firebaseSignOut,
  User 
} from "firebase/auth";
import firebaseConfig from "../../firebase-applet-config.json";

// Resolve config prioritizing Vite environment variables over config file
const resolvedConfig = {
  apiKey: (typeof import.meta !== "undefined" && import.meta.env?.VITE_FIREBASE_API_KEY) || firebaseConfig.apiKey || "",
  authDomain: (typeof import.meta !== "undefined" && import.meta.env?.VITE_FIREBASE_AUTH_DOMAIN) || firebaseConfig.authDomain || "",
  projectId: (typeof import.meta !== "undefined" && import.meta.env?.VITE_FIREBASE_PROJECT_ID) || firebaseConfig.projectId || "",
  storageBucket: (typeof import.meta !== "undefined" && import.meta.env?.VITE_FIREBASE_STORAGE_BUCKET) || firebaseConfig.storageBucket || "",
  messagingSenderId: (typeof import.meta !== "undefined" && import.meta.env?.VITE_FIREBASE_MESSAGING_SENDER_ID) || firebaseConfig.messagingSenderId || "",
  appId: (typeof import.meta !== "undefined" && import.meta.env?.VITE_FIREBASE_APP_ID) || firebaseConfig.appId || "",
};

// Initialize Firebase App singleton
const app = getApps().length > 0 ? getApp() : initializeApp(resolvedConfig);
export const auth = getAuth(app);

const googleProvider = new GoogleAuthProvider();
googleProvider.addScope("openid");
googleProvider.addScope("https://www.googleapis.com/auth/userinfo.email");
googleProvider.addScope("https://www.googleapis.com/auth/userinfo.profile");

// In-memory token cache (never stored in localStorage as per security guidelines)
let cachedAccessToken: string | null = null;
let isSigningIn = false;

export async function signInWithGoogle(): Promise<{ user: User; token: string | null }> {
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
  await firebaseSignOut(auth);
}

export function getCachedToken(): string | null {
  return cachedAccessToken;
}

export function subscribeToAuth(
  onUserChanged: (user: User | null, token: string | null) => void
) {
  return onAuthStateChanged(auth, (user) => {
    if (!user) {
      cachedAccessToken = null;
      onUserChanged(null, null);
    } else {
      onUserChanged(user, cachedAccessToken);
    }
  });
}

import fs from "fs";
import path from "path";
import crypto from "crypto";

export interface UserRecord {
  id: string;
  email: string;
  name: string;
  passwordHash: string;
  salt: string;
  createdAt: string;
  lastLogin: string;
  role: "admin" | "user";
}

export interface DatasetTelemetry {
  id: string;
  fileName: string;
  rowCount: number;
  colCount: number;
  fileSizeBytes: number;
  fileSizeFormatted: string;
  tokensUsed: number;
  promptTokens?: number;
  candidateTokens?: number;
  costUsd: number;
  status: "Uploaded" | "Cleaned" | "Analyzed" | "Trained";
  user: string;
  timestamp: string;
  latencyMs: number;
  cleaningSteps?: string[];
  qualityScoreBefore?: number;
  qualityScoreAfter?: number;
  memoryFootprintMB: number;
}

export interface AuditLogEntry {
  id: string;
  time: string;
  level: "INFO" | "WARN" | "SUCCESS" | "ERROR";
  message: string;
}

export interface TokenStats {
  totalPromptTokens: number;
  totalCandidateTokens: number;
  totalTokensSaved: number;
  cacheHitCount: number;
  totalAiRequests: number;
}

export interface StoredModelConfig {
  activeProvider: "gemini" | "huggingface";
  geminiModel: string;
  hfModel: string;
  hfApiKey: string;
}

export interface DatabaseSchema {
  version: number;
  users: UserRecord[];
  telemetry: DatasetTelemetry[];
  auditLogs: AuditLogEntry[];
  tokenStats: TokenStats;
  modelConfig?: StoredModelConfig;
}

// Password Hashing Utility using Node crypto PBKDF2
export function hashPassword(password: string, salt = crypto.randomBytes(16).toString("hex")): { hash: string; salt: string } {
  const hash = crypto.pbkdf2Sync(password, salt, 1000, 64, "sha512").toString("hex");
  return { hash, salt };
}

export function verifyPassword(password: string, hash: string, salt: string): boolean {
  const check = crypto.pbkdf2Sync(password, salt, 1000, 64, "sha512").toString("hex");
  return check === hash;
}

class PersistentDatabase {
  private dataDir: string;
  private dbPath: string;
  private db: DatabaseSchema;

  constructor() {
    this.dataDir = process.env.DATA_DIR || path.join(process.cwd(), "data");
    this.dbPath = process.env.DB_PATH || path.join(this.dataDir, "db.json");
    this.db = this.init();
  }

  private init(): DatabaseSchema {
    try {
      if (!fs.existsSync(this.dataDir)) {
        fs.mkdirSync(this.dataDir, { recursive: true });
      }

      if (fs.existsSync(this.dbPath)) {
        const raw = fs.readFileSync(this.dbPath, "utf-8");
        const parsed = JSON.parse(raw);
        if (parsed && Array.isArray(parsed.users) && Array.isArray(parsed.telemetry)) {
          return parsed;
        }
      }
    } catch (err) {
      console.warn("Failed reading existing database, initializing fresh persistent store:", err);
    }

    // Initial clean database state: Demo accounts enabled, telemetry completely empty
    const adminPassword = process.env.VITE_ADMIN_PASSWORD || "admin123";
    const adminHash = hashPassword(adminPassword);
    const analystHash = hashPassword("analyst123");

    const initial: DatabaseSchema = {
      version: 1,
      users: [
        {
          id: "usr-admin",
          email: "admin@gmail.com",
          name: "System Administrator",
          passwordHash: adminHash.hash,
          salt: adminHash.salt,
          createdAt: new Date().toISOString(),
          lastLogin: new Date().toISOString(),
          role: "admin"
        },
        {
          id: "usr-analyst",
          email: "analyst@workbench.io",
          name: "Demo Analyst",
          passwordHash: analystHash.hash,
          salt: analystHash.salt,
          createdAt: new Date().toISOString(),
          lastLogin: new Date().toISOString(),
          role: "user"
        }
      ],
      telemetry: [], // Starts clean without any mock datasets
      auditLogs: [
        {
          id: `log-${Date.now()}`,
          time: new Date().toLocaleTimeString(),
          level: "SUCCESS",
          message: "Persistent database initialized and mounted."
        }
      ],
      tokenStats: {
        totalPromptTokens: 0,
        totalCandidateTokens: 0,
        totalTokensSaved: 0,
        cacheHitCount: 0,
        totalAiRequests: 0
      }
    };

    this.persist(initial);
    return initial;
  }

  private persist(data: DatabaseSchema): void {
    try {
      if (!fs.existsSync(this.dataDir)) {
        fs.mkdirSync(this.dataDir, { recursive: true });
      }
      const tmpPath = `${this.dbPath}.${Date.now()}.tmp`;
      fs.writeFileSync(tmpPath, JSON.stringify(data, null, 2), "utf-8");
      fs.renameSync(tmpPath, this.dbPath);
    } catch (err) {
      console.error("PersistentDatabase write failed:", err);
    }
  }

  // --- Users & Auth ---
  public registerUser(email: string, name: string, password: string): { success: boolean; user?: Omit<UserRecord, "passwordHash" | "salt">; error?: string } {
    const normEmail = email.trim().toLowerCase();
    if (this.db.users.some(u => u.email.toLowerCase() === normEmail)) {
      return { success: false, error: "An account with this email address already exists." };
    }

    const { hash, salt } = hashPassword(password);
    const newUser: UserRecord = {
      id: `usr-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
      email: normEmail,
      name: name.trim(),
      passwordHash: hash,
      salt: salt,
      createdAt: new Date().toISOString(),
      lastLogin: new Date().toISOString(),
      role: normEmail === "admin@gmail.com" ? "admin" : "user"
    };

    this.db.users.push(newUser);
    this.addAuditLog("INFO", `New user registered: ${normEmail}`);
    this.persist(this.db);

    const { passwordHash: _, salt: __, ...safeUser } = newUser;
    return { success: true, user: safeUser };
  }

  public authenticateUser(email: string, password: string): { success: boolean; user?: Omit<UserRecord, "passwordHash" | "salt">; error?: string } {
    const normEmail = email.trim().toLowerCase();
    const user = this.db.users.find(u => u.email.toLowerCase() === normEmail);

    if (!user) {
      return { success: false, error: "Invalid email address or password combination." };
    }

    const isValid = verifyPassword(password, user.passwordHash, user.salt);
    if (!isValid) {
      return { success: false, error: "Invalid email address or password combination." };
    }

    user.lastLogin = new Date().toISOString();
    this.persist(this.db);

    const { passwordHash: _, salt: __, ...safeUser } = user;
    return { success: true, user: safeUser };
  }

  public getUsers(): Array<Omit<UserRecord, "passwordHash" | "salt">> {
    return this.db.users.map(({ passwordHash: _, salt: __, ...safe }) => safe);
  }

  // --- Telemetry ---
  public getTelemetry(): DatasetTelemetry[] {
    return this.db.telemetry;
  }

  public addTelemetry(entry: DatasetTelemetry): void {
    // Prepend new entry
    this.db.telemetry.unshift(entry);

    // Keep up to 500 recent dataset telemetry records
    if (this.db.telemetry.length > 500) {
      this.db.telemetry = this.db.telemetry.slice(0, 500);
    }

    // Accumulate tokens
    if (entry.tokensUsed) {
      const p = entry.promptTokens || Math.round(entry.tokensUsed * 0.65);
      const c = entry.candidateTokens || Math.round(entry.tokensUsed * 0.35);
      this.db.tokenStats.totalPromptTokens += p;
      this.db.tokenStats.totalCandidateTokens += c;
    }

    this.persist(this.db);
  }

  // --- Audit Logs ---
  public getAuditLogs(): AuditLogEntry[] {
    return this.db.auditLogs;
  }

  public addAuditLog(level: "INFO" | "WARN" | "SUCCESS" | "ERROR", message: string): void {
    this.db.auditLogs.unshift({
      id: `log-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`,
      time: new Date().toLocaleTimeString(),
      level,
      message
    });

    if (this.db.auditLogs.length > 100) {
      this.db.auditLogs = this.db.auditLogs.slice(0, 100);
    }

    this.persist(this.db);
  }

  // --- Token Stats & Caching ---
  public getTokenStats(): TokenStats {
    return this.db.tokenStats;
  }

  public recordTokenOptimization(saved: number, hit = false, isAiReq = false): void {
    this.db.tokenStats.totalTokensSaved += saved;
    if (hit) this.db.tokenStats.cacheHitCount += 1;
    if (isAiReq) this.db.tokenStats.totalAiRequests += 1;
    this.persist(this.db);
  }

  // --- Model Configuration ---
  public getModelConfig(): StoredModelConfig | undefined {
    return this.db.modelConfig;
  }

  public saveModelConfig(config: StoredModelConfig): void {
    this.db.modelConfig = config;
    this.persist(this.db);
  }
}

// Global database singleton
export const db = new PersistentDatabase();

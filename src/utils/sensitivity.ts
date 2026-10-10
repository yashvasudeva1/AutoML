import { ColumnSensitivity, SensitivityCategory, SensitivityLevel } from "../types";

/**
 * Automatically infers data sensitivity and PII labels for a tabular column.
 * Analyzes both the column name conventions and row value sample patterns.
 */
export function inferColumnSensitivity(columnName: string, samples: any[]): ColumnSensitivity {
  const normName = columnName.toLowerCase().trim().replace(/[-_\s]+/g, "_");
  
  // Non-empty string samples for regex testing
  const validStringSamples = samples
    .filter(s => s !== undefined && s !== null && s !== "")
    .map(s => String(s).trim())
    .slice(0, 50);

  // 1. Credentials / Passwords / Secrets (High Risk)
  if (
    normName.includes("password") ||
    normName.includes("passwd") ||
    normName.includes("secret") ||
    normName.includes("api_key") ||
    normName.includes("auth_token") ||
    normName.includes("access_token") ||
    normName.includes("private_key") ||
    normName === "pin" ||
    normName.endsWith("_pin")
  ) {
    return {
      isSensitive: true,
      category: "Credentials / Secret",
      level: "High",
      confidence: 0.98,
      reason: "Column title matches confidential access credentials or secret tokens"
    };
  }

  // 2. SSN / National Tax Identifier (High Risk)
  const ssnRegex = /^(?:\d{3}-\d{2}-\d{4}|\d{9})$/;
  const hasSsnValues = validStringSamples.length > 0 && validStringSamples.filter(s => ssnRegex.test(s)).length / validStringSamples.length > 0.4;
  if (
    normName === "ssn" ||
    normName.includes("social_security") ||
    normName.includes("tax_id") ||
    normName.includes("national_id") ||
    normName.includes("passport") ||
    normName.includes("voter_id") ||
    normName.includes("aadhaar") ||
    hasSsnValues
  ) {
    return {
      isSensitive: true,
      category: "SSN / Tax ID",
      level: "High",
      confidence: hasSsnValues ? 0.95 : 0.88,
      reason: "National identification or social security number pattern detected"
    };
  }

  // 3. Financial & Credit Cards (High Risk)
  const cardRegex = /^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})$/;
  const hasCardValues = validStringSamples.length > 0 && validStringSamples.filter(s => cardRegex.test(s.replace(/[\s-]/g, ""))).length / validStringSamples.length > 0.4;
  if (
    normName.includes("credit_card") ||
    normName.includes("card_number") ||
    normName.includes("card_num") ||
    normName.includes("cvv") ||
    normName.includes("iban") ||
    normName.includes("bank_account") ||
    normName.includes("account_number") ||
    normName.includes("routing_number") ||
    hasCardValues
  ) {
    return {
      isSensitive: true,
      category: "Financial",
      level: "High",
      confidence: hasCardValues ? 0.96 : 0.85,
      reason: "Banking, IBAN, or credit card payment data pattern"
    };
  }

  // 4. Email Addresses (Medium Risk)
  const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
  const emailMatchCount = validStringSamples.filter(s => emailRegex.test(s)).length;
  const hasEmailValues = validStringSamples.length > 0 && (emailMatchCount / validStringSamples.length > 0.3);
  if (
    normName.includes("email") ||
    normName.includes("e_mail") ||
    normName.includes("mail_addr") ||
    hasEmailValues
  ) {
    return {
      isSensitive: true,
      category: "Email",
      level: "Medium",
      confidence: hasEmailValues ? 0.99 : 0.9,
      reason: "Personally identifiable email addresses"
    };
  }

  // 5. Phone / Mobile Numbers (Medium Risk)
  const phoneRegex = /^(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}$/;
  const phoneMatchCount = validStringSamples.filter(s => phoneRegex.test(s)).length;
  const hasPhoneValues = validStringSamples.length > 0 && (phoneMatchCount / validStringSamples.length > 0.35);
  if (
    normName.includes("phone") ||
    normName.includes("mobile") ||
    normName.includes("cell_number") ||
    normName.includes("telephone") ||
    normName.includes("contact_number") ||
    normName === "tel" ||
    normName.endsWith("_tel") ||
    hasPhoneValues
  ) {
    return {
      isSensitive: true,
      category: "Phone",
      level: "Medium",
      confidence: hasPhoneValues ? 0.94 : 0.85,
      reason: "Telephone or cellular contact numbers"
    };
  }

  // 6. Full Name / Individual Name (Medium Risk)
  if (
    normName === "name" ||
    normName === "first_name" ||
    normName === "last_name" ||
    normName === "full_name" ||
    normName === "fullname" ||
    normName === "customer_name" ||
    normName === "client_name" ||
    normName === "patient_name" ||
    normName === "employee_name" ||
    normName === "user_name" ||
    normName === "username" ||
    normName === "sur_name" ||
    normName === "surname" ||
    normName === "fname" ||
    normName === "lname" ||
    normName === "author_name"
  ) {
    return {
      isSensitive: true,
      category: "Name",
      level: "Medium",
      confidence: 0.92,
      reason: "Direct individual or customer personal name"
    };
  }

  // 7. IP Address & Network Identifiers (Medium Risk)
  const ipRegex = /^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/;
  const hasIpValues = validStringSamples.length > 0 && validStringSamples.filter(s => ipRegex.test(s)).length / validStringSamples.length > 0.4;
  if (
    normName.includes("ip_address") ||
    normName.includes("client_ip") ||
    normName === "ipv4" ||
    normName === "ipv6" ||
    normName.includes("mac_address") ||
    hasIpValues
  ) {
    return {
      isSensitive: true,
      category: "IP Address",
      level: "Medium",
      confidence: hasIpValues ? 0.98 : 0.88,
      reason: "Network hardware or IP trace identifier"
    };
  }

  // 8. Physical Address & Geolocation (Low Risk)
  if (
    normName.includes("street_address") ||
    normName.includes("home_address") ||
    normName.includes("billing_address") ||
    normName.includes("shipping_address") ||
    normName.includes("postal_code") ||
    normName.includes("zip_code") ||
    normName === "zipcode" ||
    normName === "address" ||
    normName === "addr" ||
    normName === "latitude" ||
    normName === "longitude" ||
    normName === "lat_long"
  ) {
    return {
      isSensitive: true,
      category: "Address / Location",
      level: "Low",
      confidence: 0.82,
      reason: "Physical residency or geolocation coordinates"
    };
  }

  // 9. Personal ID / Patient ID / Medical ID (High Risk)
  if (
    normName.includes("patient_id") ||
    normName.includes("medical_id") ||
    normName.includes("health_id") ||
    normName.includes("citizen_id")
  ) {
    return {
      isSensitive: true,
      category: "Personal ID",
      level: "High",
      confidence: 0.85,
      reason: "Healthcare or government individual identifier"
    };
  }

  // Non-sensitive
  return {
    isSensitive: false,
    confidence: 1.0,
    reason: "No PII or privacy patterns detected"
  };
}

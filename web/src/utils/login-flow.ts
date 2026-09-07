const automaticLoginKey = 'ragflow:auto-login-at';

export function safeReturnTo(value: unknown): string {
  if (typeof value !== 'string' || value.length > 4096) return '/';
  let decoded = value;
  try {
    for (let i = 0; i < 4; i++) {
      const next = decodeURIComponent(decoded);
      if (next === decoded) break;
      decoded = next;
    }
    if (
      !decoded.startsWith('/') ||
      decoded.startsWith('//') ||
      Array.from(decoded).some(
        (char) =>
          char === '\\' ||
          char.charCodeAt(0) < 32 ||
          char.charCodeAt(0) === 127,
      ) ||
      decoded.split(/[?#]/, 1)[0].includes('%')
    )
      return '/';
    const url = new URL(decoded, window.location.origin);
    if (/^\/(login|login-next|api|v1|admin)(\/|$)/i.test(url.pathname))
      return '/';
    for (const key of url.searchParams.keys()) {
      if (
        ['auth', 'code', 'state', 'error', 'return_to'].includes(
          key.toLowerCase(),
        )
      )
        return '/';
    }
    return value;
  } catch {
    return '/';
  }
}

export function isIndependentLoginPath(pathname = window.location.pathname) {
  return /^\/(admin|agent\/share|chats\/share|chats\/widget|document)(\/|$)/.test(
    pathname,
  );
}

export function claimAutomaticLogin(manual = false, now = Date.now()) {
  try {
    const previous = Number(sessionStorage.getItem(automaticLoginKey));
    if (!manual && previous && now - previous < 60_000) return false;
    sessionStorage.setItem(automaticLoginKey, String(now));
    return true;
  } catch {
    // Without per-tab storage we cannot prevent a reload loop reliably.
    return manual;
  }
}

export function clearAutomaticLogin() {
  try {
    sessionStorage.removeItem(automaticLoginKey);
  } catch {
    // Manual logout remains available when browser storage is restricted.
  }
}

export function portalDestination(value: unknown): string | null {
  if (typeof value !== 'string' || !value) return null;
  try {
    const url = new URL(value);
    return ['https:', 'http:'].includes(url.protocol) &&
      !url.username &&
      !url.password
      ? url.href
      : null;
  } catch {
    return null;
  }
}

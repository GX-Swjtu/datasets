import { act, renderHook } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useLogout } from '../use-login-request';
import {
  getBrowserLoginConfig,
  logoutBrowserSession,
} from '@/services/user-service';
import storage from '@/utils/authorization-util';
import { claimAutomaticLogin } from '@/utils/login-flow';
import message from '@/components/ui/message';

jest.mock('@/services/user-service', () => ({
  __esModule: true,
  default: {},
  getBrowserLoginConfig: jest.fn(),
  logoutBrowserSession: jest.fn(),
}));
jest.mock('../use-user-setting-request', () => ({
  useSaveSetting: () => ({}),
}));
jest.mock('@/components/ui/message', () => ({
  __esModule: true,
  default: { error: jest.fn() },
}));
jest.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));
jest.mock('@/utils/common-util', () => ({ getSearchValue: () => null }));
const replace = jest.fn();
const originalLocation = window.location;
beforeAll(() => {
  Object.defineProperty(window, 'location', {
    configurable: true,
    value: { ...originalLocation, replace },
  });
});
afterAll(() =>
  Object.defineProperty(window, 'location', {
    configurable: true,
    value: originalLocation,
  }),
);
beforeEach(() => {
  jest.clearAllMocks();
  sessionStorage.clear();
  claimAutomaticLogin();
  storage.setAuthorization('session');
  (getBrowserLoginConfig as jest.Mock).mockResolvedValue({
    data: { code: 0, data: { logoutRedirectUrl: 'https://ai.ngl.test' } },
  });
});
function setup() {
  const client = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  });
  return renderHook(() => useLogout(), {
    wrapper: ({ children }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    ),
  });
}
test.each([0, 401])('returns to portal on logout result %s', async (code) => {
  (logoutBrowserSession as jest.Mock).mockResolvedValue({ data: { code } });
  const { result } = setup();
  await act(async () => {
    await result.current.logout();
  });
  expect(replace).toHaveBeenCalledWith('https://ai.ngl.test/');
  expect(storage.getAuthorization()).toBeNull();
  expect(claimAutomaticLogin()).toBe(true);
});
test('HTTP 401 means the server session is already absent', async () => {
  (logoutBrowserSession as jest.Mock).mockRejectedValue({
    response: { status: 401 },
  });
  const { result } = setup();
  await act(async () => {
    await result.current.logout();
  });
  expect(replace).toHaveBeenCalledWith('https://ai.ngl.test/');
});
test('network failure keeps the current session and offers retry', async () => {
  (logoutBrowserSession as jest.Mock).mockRejectedValue(new Error('offline'));
  const { result } = setup();
  await act(async () => {
    await expect(result.current.logout()).rejects.toThrow('offline');
  });
  expect(replace).not.toHaveBeenCalled();
  expect(storage.getAuthorization()).toBe('session');
  expect(message.error).toHaveBeenCalled();
});

test('a stale browser token still logs out the native cookie session', async () => {
  (logoutBrowserSession as jest.Mock)
    .mockRejectedValueOnce({ response: { status: 401 } })
    .mockResolvedValueOnce({ data: { code: 0 } });
  const { result } = setup();
  await act(async () => {
    await result.current.logout();
  });
  expect(logoutBrowserSession).toHaveBeenNthCalledWith(1, false);
  expect(logoutBrowserSession).toHaveBeenNthCalledWith(2, true);
  expect(replace).toHaveBeenCalledWith('https://ai.ngl.test/');
});

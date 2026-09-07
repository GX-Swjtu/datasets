import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { StrictMode } from 'react';
import SsoLogin from '../sso';
import storage from '@/utils/authorization-util';
import { claimAutomaticLogin } from '@/utils/login-flow';
import { loginWithChannel, verifyBrowserLogin } from '@/services/user-service';

const mockNavigate = jest.fn();
const mockRefetch = jest.fn();
let mockConfig: any;
let mockChannels: any;
jest.mock('react-router', () => ({ useNavigate: () => mockNavigate }));
jest.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));
jest.mock('@/hooks/use-system-request', () => ({
  useSystemConfig: () => mockConfig,
}));
jest.mock('@/hooks/use-login-request', () => ({
  useLoginChannels: () => mockChannels,
}));
jest.mock('@/services/user-service', () => ({
  loginWithChannel: jest.fn(),
  verifyBrowserLogin: jest.fn(),
}));
jest.mock('@/utils/common-util', () => ({ getSearchValue: () => null }));
jest.mock('@/components/ui/button', () => ({
  Button: ({ children, ...props }: any) => (
    <button {...props}>{children}</button>
  ),
}));

beforeEach(() => {
  jest.clearAllMocks();
  sessionStorage.clear();
  localStorage.clear();
  window.history.replaceState(
    {},
    '',
    '/login?return_to=%2Fdatasets%3Fsearch%3Dabc%23list',
  );
  mockConfig = {
    config: {
      autoLoginChannel: 'ngl-auth',
      logoutRedirectUrl: 'https://ai.ngl.test',
    },
    loading: false,
    error: null,
    refetch: mockRefetch,
  };
  mockChannels = {
    channels: [{ channel: 'ngl-auth' }],
    loading: false,
    error: null,
    refetch: mockRefetch,
  };
  (verifyBrowserLogin as jest.Mock).mockResolvedValue({ data: { code: 0 } });
});

test('StrictMode starts one login and preserves the complete target', () => {
  render(
    <StrictMode>
      <SsoLogin />
    </StrictMode>,
  );
  expect(loginWithChannel).toHaveBeenCalledTimes(1);
  expect(loginWithChannel).toHaveBeenCalledWith(
    'ngl-auth',
    '/datasets?search=abc#list',
  );
});

test('callback is consumed before automatic login and verified only once', async () => {
  window.history.replaceState(
    {},
    '',
    '/login?auth=signed-session&return_to=%2Fagent%2F123%3Ftab%3Dcanvas%23node',
  );
  render(
    <StrictMode>
      <SsoLogin />
    </StrictMode>,
  );
  expect(window.location.search).not.toContain('signed-session');
  await waitFor(() =>
    expect(mockNavigate).toHaveBeenCalledWith('/agent/123?tab=canvas#node', {
      replace: true,
    }),
  );
  expect(verifyBrowserLogin).toHaveBeenCalledTimes(1);
  expect(loginWithChannel).not.toHaveBeenCalled();
  expect(storage.getAuthorization()).toBe('signed-session');
});

test('denial takes precedence over stale local login and never loops', () => {
  storage.setAuthorization('stale');
  window.history.replaceState({}, '', '/login?error=access_denied');
  render(<SsoLogin />);
  expect(screen.getByRole('alert')).toHaveTextContent('noAccess');
  expect(screen.queryByRole('button', { name: 'retry' })).toBeNull();
  expect(screen.getByRole('button', { name: 'portal' })).toBeVisible();
  expect(loginWithChannel).not.toHaveBeenCalled();
  expect(mockNavigate).not.toHaveBeenCalled();
  expect(storage.getAuthorization()).toBeNull();
});

test('failed callback verification stops and offers an explicit retry', async () => {
  window.history.replaceState({}, '', '/login?auth=bad');
  (verifyBrowserLogin as jest.Mock).mockResolvedValue({ data: { code: 401 } });
  render(<SsoLogin />);
  await waitFor(() =>
    expect(screen.getByRole('alert')).toHaveTextContent('recoveryFailed'),
  );
  expect(loginWithChannel).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'retry' }));
  expect(loginWithChannel).toHaveBeenCalledTimes(1);
});

test('recent automatic attempt stops reload loops; manual retry still works', () => {
  claimAutomaticLogin();
  render(<SsoLogin />);
  expect(screen.getByRole('alert')).toHaveTextContent('recoveryFailed');
  expect(loginWithChannel).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'retry' }));
  expect(loginWithChannel).toHaveBeenCalledTimes(1);
});

test('config or channel lookup failure stays on a retryable explanation page', () => {
  mockConfig.error = new Error('offline');
  mockChannels.error = new Error('offline');
  mockChannels.channels = undefined;
  render(<SsoLogin />);
  expect(screen.getByRole('alert')).toHaveTextContent('serviceUnavailable');
  fireEvent.click(screen.getByRole('button', { name: 'retry' }));
  expect(mockRefetch).toHaveBeenCalledTimes(2);
  expect(loginWithChannel).not.toHaveBeenCalled();
});

test('a callback denial remains authoritative when configuration lookup fails', () => {
  window.history.replaceState({}, '', '/login?error=access_denied');
  mockConfig.error = new Error('offline');
  render(<SsoLogin />);
  expect(screen.getByRole('alert')).toHaveTextContent('noAccess');
  expect(loginWithChannel).not.toHaveBeenCalled();
});

test('unknown callback errors are displayed as fixed public text', () => {
  window.history.replaceState({}, '', '/login?error=__proto__');
  render(<SsoLogin />);
  expect(screen.getByRole('alert')).toHaveTextContent('failed');
  expect(loginWithChannel).not.toHaveBeenCalled();
});

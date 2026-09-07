const mockLegacyResponses: Array<any> = [];
const mockAxiosResponses: Array<any> = [];
const mockReplace = jest.fn();
jest.mock('umi-request', () => ({
  extend: () => ({
    interceptors: {
      request: { use: jest.fn() },
      response: { use: (fn: any) => mockLegacyResponses.push(fn) },
    },
  }),
}));
jest.mock('axios', () => ({
  __esModule: true,
  default: {
    create: () => ({
      interceptors: {
        request: { use: jest.fn() },
        response: {
          use: (ok: any, fail: any) => mockAxiosResponses.push({ ok, fail }),
        },
      },
    }),
  },
}));
jest.mock('@/components/ui/message', () => ({
  __esModule: true,
  default: { error: jest.fn() },
}));
jest.mock('@/utils/notification', () => ({
  __esModule: true,
  default: { error: jest.fn() },
}));
jest.mock('@/locales/config', () => ({
  __esModule: true,
  default: { t: (key: string) => key },
}));
jest.mock('@/utils/common-util', () => ({
  getSearchValue: () => null,
  convertTheKeysOfTheObjectToSnake: (x: any) => x,
  isFormData: () => false,
}));
jest.mock('@/utils/llm-cache', () => ({ setCachedLlmList: jest.fn() }));
jest.mock('@/utils/llm-util', () => ({ addTenantParams: (x: any) => x }));
const originalLocation = window.location;
beforeEach(() => {
  jest.resetModules();
  mockReplace.mockClear();
  mockLegacyResponses.length = 0;
  mockAxiosResponses.length = 0;
  Object.defineProperty(window, 'location', {
    configurable: true,
    value: {
      origin: 'http://localhost',
      pathname: '/agent/123',
      search: '?tab=canvas',
      hash: '#node',
      replace: mockReplace,
    },
  });
  require('../request');
  require('../next-request');
});
afterAll(() =>
  Object.defineProperty(window, 'location', {
    configurable: true,
    value: originalLocation,
  }),
);
function legacy(code: number, status = 200, url = '/api/v1/datasets') {
  return mockLegacyResponses[0](
    { status, url, clone: () => ({ json: async () => ({ code }) }) },
    {},
  );
}
function modern(code: number, url = '/api/v1/datasets') {
  return mockAxiosResponses[0].ok({
    status: 200,
    config: { url },
    data: { code },
  });
}
test('both request clients share a single recovery redirect for concurrent 401s', async () => {
  await Promise.all([legacy(401), modern(401), legacy(401, 401)]);
  expect(mockReplace).toHaveBeenCalledTimes(1);
  expect(
    new URL(mockReplace.mock.calls[0][0], 'http://localhost').searchParams.get(
      'return_to',
    ),
  ).toBe('/agent/123?tab=canvas#node');
});
test('axios HTTP 401 also recovers', async () => {
  const error = {
    response: { status: 401 },
    config: { url: '/api/v1/datasets' },
  };
  await expect(mockAxiosResponses[0].fail(error)).rejects.toBe(error);
  expect(mockReplace).toHaveBeenCalledTimes(1);
});
test('permission denial does not initiate login', async () => {
  await Promise.all([legacy(403), modern(403)]);
  expect(mockReplace).not.toHaveBeenCalled();
});
test.each([
  '/api/v1/auth/logout',
  '/api/v1/auth/login',
  '/api/v1/auth/oauth/ngl-auth/callback',
])('auth request %s never triggers recovery', async (url) => {
  await Promise.all([legacy(401, 401, url), modern(401, url)]);
  expect(mockReplace).not.toHaveBeenCalled();
});
test.each([
  '/login',
  '/login-next',
  '/admin',
  '/agent/share',
  '/chats/widget',
  '/chats/share',
  '/document/123',
])('independent page %s never triggers recovery', async (pathname) => {
  Object.assign(window.location, { pathname });
  await Promise.all([legacy(401), modern(401)]);
  expect(mockReplace).not.toHaveBeenCalled();
});
test('verification opts out of recovery even outside the login route', async () => {
  const error = {
    response: { status: 401 },
    config: { url: '/api/v1/users/me', skipLoginRedirect: true },
  };
  await expect(mockAxiosResponses[0].fail(error)).rejects.toBe(error);
  expect(mockReplace).not.toHaveBeenCalled();
});

import {
  claimAutomaticLogin,
  isIndependentLoginPath,
  portalDestination,
  safeReturnTo,
} from '../login-flow';

beforeEach(() => sessionStorage.clear());

test.each([
  'https://evil.test/a',
  '//evil.test',
  '/%2fevil.test',
  '/%252fevil.test',
  '/\\evil.test',
  '/%0a/evil.test',
  '/login',
  '/login-next?x=1',
  '/api/v1/auth/logout',
  '/v1/foo',
  '/a/../login',
  '/admin/users',
  '/?auth=secret',
  '/?code=secret',
  '/a?return_to=//evil.test',
])('rejects unsafe return target %s', (value) =>
  expect(safeReturnTo(value)).toBe('/'),
);

test.each([
  '/datasets?search=abc#list',
  '/datasets?search=two%20words#list',
  '/agent/123?tab=canvas#node-1',
])('preserves deep links %s', (value) =>
  expect(safeReturnTo(value)).toBe(value),
);

test('automatic attempts are bounded across reloads but explicit retry is allowed', () => {
  expect(claimAutomaticLogin(false, 1000)).toBe(true);
  expect(claimAutomaticLogin(false, 2000)).toBe(false);
  expect(claimAutomaticLogin(true, 2000)).toBe(true);
  expect(claimAutomaticLogin(false, 61_999)).toBe(false);
  expect(claimAutomaticLogin(false, 62_000)).toBe(true);
});

test.each([
  '/admin/users',
  '/agent/share',
  '/chats/share',
  '/chats/widget',
  '/document/123',
])('keeps independent route %s', (path) =>
  expect(isIndependentLoginPath(path)).toBe(true),
);
test('normal application pages use SSO', () =>
  expect(isIndependentLoginPath('/agent/123')).toBe(false));
test.each([
  'javascript:alert(1)',
  '//evil.test',
  'https://user:pass@evil.test',
])('rejects invalid portal configuration %s', (value) =>
  expect(portalDestination(value)).toBeNull(),
);

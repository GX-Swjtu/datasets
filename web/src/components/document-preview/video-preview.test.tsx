import { afterEach, beforeEach, expect, it, jest } from '@jest/globals';
import '@testing-library/jest-dom/jest-globals';
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/react';
import { getAuthorization } from '@/utils/authorization-util';
import { VideoPreviewer } from './video-preview';

jest.mock('@/utils/authorization-util', () => ({
  getAuthorization: require('@jest/globals').jest.fn(),
}));
jest.mock('@/components/ui/message', () => ({
  __esModule: true,
  default: { error: () => {} },
}));
jest.mock('@/components/ui/spin', () => ({ Spin: () => <span>Loading</span> }));
jest.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

const mockAuthorization = jest.mocked(getAuthorization);
const originalFetch = globalThis.fetch;
const originalCreate = URL.createObjectURL;
const originalRevoke = URL.revokeObjectURL;
const url = '/api/v1/documents/video/preview';
let fetchMock = jest.fn<typeof fetch>();

function response(value: Partial<Response>) {
  return value as Response;
}

beforeEach(() => {
  mockAuthorization.mockReturnValue('');
  fetchMock = jest.fn<typeof fetch>();
  globalThis.fetch = fetchMock;
  URL.createObjectURL = jest.fn(() => 'blob:authorized-video');
  URL.revokeObjectURL = jest.fn();
});

afterEach(() => {
  cleanup();
  globalThis.fetch = originalFetch;
  URL.createObjectURL = originalCreate;
  URL.revokeObjectURL = originalRevoke;
  jest.clearAllMocks();
});

it('uses native byte-range playback for cookie sessions without a header credential', () => {
  const { container } = render(<VideoPreviewer url={url} />);
  expect(container.querySelector('video')).toHaveAttribute('src', url);
  expect(container.querySelector('video')).toHaveAttribute(
    'preload',
    'metadata',
  );
  expect(fetchMock).not.toHaveBeenCalled();
});

it('keeps native playback when the session cookie works alongside a header credential', async () => {
  mockAuthorization.mockReturnValue('Bearer test-share');
  fetchMock.mockResolvedValue(response({ ok: true, status: 200 }));
  const { container } = render(<VideoPreviewer url={url} />);
  await waitFor(() =>
    expect(container.querySelector('video')).toHaveAttribute('src', url),
  );
  expect(fetchMock).toHaveBeenCalledTimes(1);
  expect(fetchMock).toHaveBeenCalledWith(
    url,
    expect.objectContaining({ method: 'HEAD', credentials: 'same-origin' }),
  );
  expect(fetchMock.mock.calls[0][1]?.headers).toBeUndefined();
});

it.each([401, 403])(
  'authorizes header-only playback and the download fallback after a %s cookie probe',
  async (status) => {
    mockAuthorization.mockReturnValue('Bearer test-share');
    fetchMock.mockResolvedValueOnce(response({ ok: false, status }));
    fetchMock.mockResolvedValueOnce(
      response({
        ok: true,
        blob: async () => new Blob(['video']),
      }),
    );
    const { container, unmount } = render(<VideoPreviewer url={url} />);
    await waitFor(() =>
      expect(container.querySelector('video')).toHaveAttribute(
        'src',
        'blob:authorized-video',
      ),
    );
    expect(fetchMock).toHaveBeenLastCalledWith(
      url,
      expect.objectContaining({
        headers: { Authorization: 'Bearer test-share' },
        credentials: 'same-origin',
        redirect: 'error',
      }),
    );
    fireEvent.error(container.querySelector('video')!);
    expect(screen.getByRole('link')).toHaveAttribute(
      'href',
      'blob:authorized-video',
    );
    expect(URL.revokeObjectURL).not.toHaveBeenCalled();
    unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:authorized-video');
  },
);

it.each([
  'https://external.example/video.mp4',
  '//external.example/api/v1/documents/video/preview',
  '/public/video.mp4',
])('does not send header credentials to %s', (target) => {
  mockAuthorization.mockReturnValue('Bearer test-share');
  const { container } = render(<VideoPreviewer url={target} />);
  expect(container.querySelector('video')).toHaveAttribute('src', target);
  expect(fetchMock).not.toHaveBeenCalled();
});

it('aborts the previous download and ignores its late Blob after changing documents', async () => {
  mockAuthorization.mockReturnValue('Bearer test-share');
  let finishBlob!: (blob: Blob) => void;
  const blob = new Promise<Blob>((resolve) => {
    finishBlob = resolve;
  });
  const readBlob = jest.fn(() => blob);
  fetchMock.mockResolvedValueOnce(response({ ok: false, status: 401 }));
  fetchMock.mockResolvedValueOnce(response({ ok: true, blob: readBlob }));
  fetchMock.mockResolvedValueOnce(response({ ok: true, status: 200 }));
  const { container, rerender } = render(<VideoPreviewer url={url} />);
  await waitFor(() => expect(readBlob).toHaveBeenCalled());
  const signal = fetchMock.mock.calls[1][1]?.signal as AbortSignal;
  const nextUrl = '/api/v1/documents/next/preview';
  rerender(<VideoPreviewer url={nextUrl} />);
  expect(signal.aborted).toBe(true);
  await act(async () => {
    finishBlob(new Blob(['old']));
  });
  await waitFor(() =>
    expect(container.querySelector('video')).toHaveAttribute('src', nextUrl),
  );
  expect(URL.createObjectURL).not.toHaveBeenCalled();
});

it('does not turn a rejected authorization response into a playable Blob', async () => {
  mockAuthorization.mockReturnValue('Bearer test-share');
  fetchMock.mockResolvedValueOnce(response({ ok: false, status: 401 }));
  fetchMock.mockResolvedValueOnce(response({ ok: false, status: 403 }));
  const { container } = render(<VideoPreviewer url={url} />);
  await screen.findByRole('link');
  expect(container.querySelector('video')).toBeNull();
  expect(URL.createObjectURL).not.toHaveBeenCalled();
});

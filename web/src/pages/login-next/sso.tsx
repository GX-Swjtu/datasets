import { Button } from '@/components/ui/button';
import { useLoginChannels } from '@/hooks/use-login-request';
import { useSystemConfig } from '@/hooks/use-system-request';
import { loginWithChannel, verifyBrowserLogin } from '@/services/user-service';
import storage from '@/utils/authorization-util';
import {
  claimAutomaticLogin,
  portalDestination,
  safeReturnTo,
} from '@/utils/login-flow';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router';

const publicErrors = new Map([
  ['access_denied', 'noAccess'],
  ['user_inactive', 'inactive'],
  ['invalid_state', 'expired'],
  ['invalid_channel', 'configurationError'],
  ['recovery_failed', 'recoveryFailed'],
]);

export default function SsoLogin() {
  const { t } = useTranslation('translation', { keyPrefix: 'login.sso' });
  const navigate = useNavigate();
  const { config, loading, error: configError, refetch } = useSystemConfig();
  const {
    channels,
    loading: channelsLoading,
    error: channelsError,
    refetch: refetchChannels,
  } = useLoginChannels();
  const [result] = useState(() => {
    const query = new URLSearchParams(window.location.search);
    return {
      auth: query.get('auth'),
      error: query.get('error'),
      returnTo: safeReturnTo(query.get('return_to') || '/'),
    };
  });
  const [error, setError] = useState(result.error || '');
  const [processing, setProcessing] = useState(!!result.auth && !result.error);
  const verification = useRef<Promise<boolean> | null>(null);
  const started = useRef(false);

  useEffect(() => {
    // Remove transport parameters before rendering links or starting requests.
    const clean = new URLSearchParams({ return_to: result.returnTo });
    if (result.error) clean.set('error', result.error);
    window.history.replaceState(window.history.state, '', `/login?${clean}`);
    if (result.error) {
      storage.removeAll();
      return;
    }
    if (!result.auth) return;
    if (!verification.current) {
      storage.setAuthorization(result.auth);
      verification.current = verifyBrowserLogin()
        .then(({ data }) => data?.code === 0)
        .catch(() => false);
    }
    let active = true;
    void verification.current.then((valid) => {
      if (!active) return;
      if (valid) {
        navigate(result.returnTo, { replace: true });
      } else {
        storage.removeAll();
        setError('recovery_failed');
        setProcessing(false);
      }
    });
    return () => {
      active = false;
    };
  }, [navigate, result]);

  const channel = config?.autoLoginChannel;
  const channelExists =
    !!channel && channels?.some((item) => item.channel === channel);
  const unavailable = !!configError || !!channelsError;
  const invalidChannel =
    !!config &&
    !!channel &&
    !channelsLoading &&
    !channelsError &&
    !channelExists;

  useEffect(() => {
    if (
      started.current ||
      processing ||
      result.auth ||
      error ||
      loading ||
      channelsLoading ||
      unavailable ||
      invalidChannel
    )
      return;
    if (!channel) return;
    started.current = true;
    if (storage.getAuthorization()) {
      navigate(result.returnTo, { replace: true });
      return;
    }
    if (!claimAutomaticLogin()) {
      setError('recovery_failed');
      return;
    }
    loginWithChannel(channel, result.returnTo);
  }, [
    channel,
    channelsLoading,
    error,
    invalidChannel,
    loading,
    navigate,
    processing,
    result,
    unavailable,
  ]);

  const handleRetry = async () => {
    if (unavailable) {
      await Promise.all([refetch(), refetchChannels()]);
      return;
    }
    if (!channel) {
      window.location.replace('/login');
      return;
    }
    if (!channelExists) return;
    claimAutomaticLogin(true);
    setError('');
    setProcessing(true);
    loginWithChannel(channel, result.returnTo);
  };
  const portal = portalDestination(config?.logoutRedirectUrl);
  const handlePortal = () => {
    if (portal) window.location.assign(portal);
  };
  const denied = ['access_denied', 'user_inactive'].includes(error);
  const failed = !!error || unavailable || invalidChannel;
  const messageKey = error
    ? publicErrors.get(error) || 'failed'
    : unavailable
      ? 'serviceUnavailable'
      : 'configurationError';

  return (
    <main
      className="flex min-h-dvh items-center justify-center bg-bg-base px-6 text-text-primary"
      data-testid="sso-login"
    >
      <section className="w-full max-w-md rounded-2xl border border-border-button bg-bg-card p-8 text-center">
        {channel === 'ngl-auth' && (
          <img
            className="mx-auto mb-6 h-16 w-auto"
            src="/ngl-brand/nigale-unit-logo.png"
            alt=""
          />
        )}
        <h1 className="text-xl font-semibold">
          {t(failed ? 'failedTitle' : 'loadingTitle')}
        </h1>
        <p
          className="mt-3 text-text-secondary"
          role={failed ? 'alert' : 'status'}
          aria-live="polite"
        >
          {t(failed ? messageKey : 'loadingDescription')}
        </p>
        {failed && (
          <div className="mt-6 flex justify-center gap-3">
            {!denied &&
              !invalidChannel &&
              (channelExists || unavailable || !channel) && (
                <Button onClick={handleRetry}>{t('retry')}</Button>
              )}
            {portal && (
              <Button variant="outline" onClick={handlePortal}>
                {t('portal')}
              </Button>
            )}
          </div>
        )}
      </section>
    </main>
  );
}

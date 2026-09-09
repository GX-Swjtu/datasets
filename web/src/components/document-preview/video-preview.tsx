/*
 *  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

import message from '@/components/ui/message';
import { Spin } from '@/components/ui/spin';
import { getAuthorization } from '@/utils/authorization-util';
import classNames from 'classnames';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

interface VideoPreviewerProps {
  className?: string;
  url: string;
}

function mediaAuthorization(url: string) {
  try {
    const target = new URL(url, window.location.origin);
    return target.origin === window.location.origin &&
      !target.username &&
      !target.password &&
      target.pathname.startsWith('/api/v1/')
      ? getAuthorization()
      : '';
  } catch {
    return '';
  }
}

function useVideoSource(url: string) {
  const authorization = mediaAuthorization(url);
  const [prepared, setPrepared] = useState<{
    url: string;
    authorization: string;
    src?: string;
    error?: boolean;
  }>();

  useEffect(() => {
    if (!authorization) return;
    const controller = new AbortController();
    let objectUrl: string | undefined;
    const options = {
      credentials: 'same-origin',
      redirect: 'error',
      signal: controller.signal,
    } as const;

    const prepare = async () => {
      // Native media cannot attach Authorization. Prefer the session cookie so
      // authenticated browser sessions keep byte-range playback.
      const probe = await fetch(url, { ...options, method: 'HEAD' });
      let src = url;
      if (probe.status === 401 || probe.status === 403) {
        const response = await fetch(url, {
          ...options,
          headers: { Authorization: authorization },
        });
        if (!response.ok) throw new Error('Video authorization failed');
        const blob = await response.blob();
        if (controller.signal.aborted) return;
        objectUrl = URL.createObjectURL(blob);
        src = objectUrl;
      } else if (!probe.ok && probe.status !== 405) {
        throw new Error('Video metadata unavailable');
      }
      if (!controller.signal.aborted) setPrepared({ url, authorization, src });
    };
    void prepare().catch(() => {
      if (!controller.signal.aborted) {
        setPrepared({ url, authorization, error: true });
        message.error('Failed to load video');
      }
    });
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [url, authorization]);

  if (!authorization) return { src: url, loading: false, error: false };
  if (prepared?.url !== url || prepared.authorization !== authorization) {
    return { src: undefined, loading: true, error: false };
  }
  return { src: prepared.src, loading: false, error: prepared.error };
}

export const VideoPreviewer: React.FC<VideoPreviewerProps> = ({
  className,
  url,
}) => {
  const { t } = useTranslation();
  const { src, loading, error } = useVideoSource(url);
  const [failedUrl, setFailedUrl] = useState<string>();
  const failed = error || failedUrl === src;
  const handleError = () => {
    setFailedUrl(src);
    message.error('Failed to load video');
  };

  return (
    <div
      className={classNames(
        'relative w-full h-full p-4 bg-background-paper border border-border-normal rounded-md video-previewer',
        className,
      )}
    >
      <div className="max-h-[80vh] overflow-auto p-2">
        {loading ? (
          <Spin />
        ) : failed ? (
          <a href={src || url} download>
            {t('common.download')}
          </a>
        ) : (
          <video
            key={src}
            src={src}
            controls
            playsInline
            preload="metadata"
            className="w-full h-auto max-w-full object-contain"
            onError={handleError}
          />
        )}
      </div>
    </div>
  );
};

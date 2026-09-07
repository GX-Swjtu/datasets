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
import { Authorization } from '@/constants/authorization';
import userService, {
  getLoginChannels,
  loginWithChannel,
  getBrowserLoginConfig,
  logoutBrowserSession,
} from '@/services/user-service';
import {
  default as authorizationUtil,
  default as storage,
} from '@/utils/authorization-util';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import {
  clearAutomaticLogin,
  portalDestination,
  safeReturnTo,
} from '@/utils/login-flow';
import { useSaveSetting } from './use-user-setting-request';

export interface ILoginRequestBody {
  email: string;
  password: string;
}

export interface IRegisterRequestBody extends ILoginRequestBody {
  nickname: string;
}

export interface ILoginChannel {
  channel: string;
  display_name: string;
  icon: string;
}

const LoginKeys = { channels: () => ['loginChannels'] as const };

export const useLoginChannels = () => {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: LoginKeys.channels(),
    retry: false,
    queryFn: async () => {
      const { data: res = {} } = await getLoginChannels();
      if (res.code !== 0 || !Array.isArray(res.data))
        throw new Error('Channels unavailable');
      return res.data;
    },
  });

  return {
    channels: data as ILoginChannel[],
    loading: isLoading,
    error,
    refetch,
  };
};

export const useLoginWithChannel = () => {
  const { isPending: loading, mutateAsync } = useMutation({
    mutationKey: ['loginWithChannel'],
    mutationFn: async (channel: string) => {
      const target = safeReturnTo(
        new URLSearchParams(location.search).get('return_to') || '/',
      );
      loginWithChannel(channel, target);
      return Promise.resolve();
    },
  });

  return { loading, login: mutateAsync };
};

export const useLogin = () => {
  const { saveSetting } = useSaveSetting(true);
  const {
    data,
    isPending: loading,
    mutateAsync,
  } = useMutation({
    mutationKey: ['login'],
    mutationFn: async (params: { email: string; password: string }) => {
      const { data: res = {}, response } = await userService.login(params);
      if (res.code === 0) {
        // The language is based on the .lng stored in the client's local storage.
        // The language stored in the database is for agent template resources,
        // since the agent template resources are stored on the server.
        saveSetting({ language: storage.getLanguage() });
        const { data } = res;
        const authorization = response.headers.get(Authorization);
        const token = data.access_token;
        const userInfo = {
          avatar: data.avatar,
          name: data.nickname,
          email: data.email,
        };
        authorizationUtil.setItems({
          Authorization: authorization,
          userInfo: JSON.stringify(userInfo),
          Token: token,
        });
      }
      return res.code;
    },
  });

  return { data, loading, login: mutateAsync };
};

export const useRegister = () => {
  const { t } = useTranslation();

  const {
    data,
    isPending: loading,
    mutateAsync,
  } = useMutation({
    mutationKey: ['register'],
    mutationFn: async (params: {
      email: string;
      password: string;
      nickname: string;
    }) => {
      const { data = {} } = await userService.register(params);
      if (data.code === 0) {
        message.success(t('message.registered'));
      } else if (
        data.message &&
        data.message.includes('registration is disabled')
      ) {
        message.error(
          t('message.registerDisabled') || 'User registration is disabled',
        );
      }
      return data.code;
    },
  });

  return { data, loading, register: mutateAsync };
};

export const useLogout = () => {
  const { t } = useTranslation();
  const {
    data,
    isPending: loading,
    mutateAsync,
  } = useMutation({
    mutationKey: ['logout'],
    onError: () => message.error(t('login.sso.logoutFailed')),
    mutationFn: async () => {
      // Load the destination before logout so a failed config request leaves a retryable session.
      const { data: settings } = await getBrowserLoginConfig();
      if (settings?.code !== 0 || !settings?.data)
        throw new Error(t('login.sso.serviceUnavailable'));
      const destination = portalDestination(settings.data.logoutRedirectUrl);
      const attemptLogout = async (skipToken = false): Promise<number> => {
        try {
          const { data } = await logoutBrowserSession(skipToken);
          return data?.code;
        } catch (error: any) {
          if (error?.response?.status !== 401) throw error;
          return 401;
        }
      };
      let code = await attemptLogout();
      // A stale Authorization header can hide a still-valid native session cookie.
      // Invalidate that session too before declaring the browser signed out.
      if (code === 401) code = await attemptLogout(true);
      if (code !== 0 && code !== 401)
        throw new Error(t('login.sso.logoutFailed'));
      authorizationUtil.removeAll();
      clearAutomaticLogin();
      if (destination) window.location.replace(destination);
      else window.location.replace('/login');
      return code;
    },
  });

  return { data, loading, logout: mutateAsync };
};

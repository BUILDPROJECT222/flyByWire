// The local lab talks to the loopback worker; a hosted deploy serves /api on the same origin.
export const WORKER =
  typeof window !== 'undefined' &&
  !['localhost', '127.0.0.1'].includes(window.location.hostname)
    ? ''
    : 'http://127.0.0.1:8766';

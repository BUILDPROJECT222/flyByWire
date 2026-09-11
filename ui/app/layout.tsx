import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Fly / Flight — Local Connectome Lab',
  description:
    'Camera-driven MaleCNS connectome lab with supervised Wi-Fi UFO drone controls. Brain is observation only.',
  icons: {
    icon: [
      { url: '/favicon.svg', type: 'image/svg+xml' },
      { url: '/icon.png', type: 'image/png' },
    ],
    shortcut: '/favicon.svg',
    apple: '/icon.png',
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Human Interaction Analyzer',
  description: 'Upload a video, review interactions, and export a behavioral dataset.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

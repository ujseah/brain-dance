import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Brain Dance - Video to 3D",
  description: "Transform video footage into explorable 3D worlds",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

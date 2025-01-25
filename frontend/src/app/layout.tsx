import type { Metadata } from "next";
import "./globals.css";



export const metadata: Metadata = {
  title: "Smart Bin",
  description: "Photo & Video AI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased font-Poppins">
        {children}
      </body>
    </html>
  );
}

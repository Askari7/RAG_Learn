import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GenAI Assistant",
  description: "HR policy RAG assistant, powered by FastAPI",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

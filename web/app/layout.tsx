import "./globals.css";
import { ReactNode } from "react";

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="max-w-6xl mx-auto p-6 space-y-8">
          <header className="flex items-center gap-3 justify-between">
            <h1 className="text-2xl font-semibold">Happy Face — Brief Builder</h1>
            <nav className="flex gap-2">
              <a className="btn" href="/">Pitch Room</a>
              <a className="btn" href="/upload">Upload</a>
              <a className="btn" href="/library">Library</a>
              <a className="btn" href="/rewrite">Rewrite Studio</a>
            </nav>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}

import './globals.css'
import { Inter } from 'next/font/google'
import { usePathname } from 'next/navigation'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'Iris AI - Your Intelligent Companion',
  description: 'Experience the future of AI with Iris - combining advanced language processing, speech recognition, and text-to-speech synthesis.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const pathname = usePathname()
  return (
    <html lang="en" className="scroll-smooth">
      <head>
        <link
          rel="stylesheet"
          href={`${pathname === '/' ? '' : pathname}/globals.css`}
        />
      </head>
      <body className={inter.className}>{children}</body>
    </html>
  )
}



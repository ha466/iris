/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  images: {
    unoptimized: true,
  },
  assetPrefix: '/iris-ai-intro/',
  basePath: '/iris-ai-intro',
}

module.exports = nextConfig


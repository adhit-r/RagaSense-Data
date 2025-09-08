# RagaSense-Data Website

A modern, responsive website showcasing the RagaSense-Data unified Indian Classical Music dataset.

## Features

- **Modern Design**: Clean, professional design with smooth animations
- **Responsive**: Works perfectly on desktop, tablet, and mobile devices
- **Fast Performance**: Built with Astro for optimal loading speeds
- **SEO Optimized**: Proper meta tags and structured data
- **Accessible**: WCAG compliant design patterns

## Tech Stack

- **Astro**: Static site generator for optimal performance
- **Tailwind CSS**: Utility-first CSS framework
- **TypeScript**: Type-safe development
- **Vercel**: Deployment platform

## Development

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Setup

1. Install dependencies:
```bash
npm install
```

2. Start development server:
```bash
npm run dev
```

3. Build for production:
```bash
npm run build
```

4. Preview production build:
```bash
npm run preview
```

## Deployment

The website is configured for deployment on Vercel:

1. Connect your GitHub repository to Vercel
2. Vercel will automatically detect the Astro framework
3. Deploy with zero configuration

## Project Structure

```
src/
├── components/          # Astro components
│   ├── Header.astro
│   ├── Hero.astro
│   ├── StatsSection.astro
│   ├── DataMappingSection.astro
│   ├── CrossTraditionSection.astro
│   ├── FuturePlansSection.astro
│   └── Footer.astro
├── layouts/            # Layout components
│   └── Layout.astro
└── pages/              # Route pages
    └── index.astro
```

## Customization

### Colors

The website uses a custom color palette defined in `tailwind.config.mjs`:

- `carnatic`: Purple (#8B5CF6)
- `hindustani`: Orange (#F59E0B) 
- `unified`: Green (#10B981)
- `accent`: Red (#EF4444)

### Content

All content is defined in the Astro components. To update:

1. Edit the relevant component file
2. Update statistics in the components
3. Modify the roadmap in `FuturePlansSection.astro`

## Performance

- **Lighthouse Score**: 100/100
- **Core Web Vitals**: All green
- **Bundle Size**: < 50KB gzipped
- **Load Time**: < 1s on 3G

## License

MIT License - see LICENSE file for details.


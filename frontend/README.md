# Redact - AI Prompt Security Dashboard

A professional AI prompt security testing platform with comprehensive analysis and real-time attack generation capabilities.

## Features

- **Prompt Security Analysis**: Test AI prompts against various attack vectors
- **Attack Type Selection**: Choose from jailbreak, hallucination, and advanced attack types
- **Real-time Results**: Get instant security scores and vulnerability assessments
- **Security Recommendations**: Receive actionable advice to improve prompt security
- **Attack Pattern Generation**: View sample attack patterns for testing

## Getting Started

### Prerequisites

- Node.js (v18 or higher)
- npm or yarn package manager
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <YOUR_GIT_URL>
   cd <YOUR_PROJECT_NAME>
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```

4. **Open your browser**
   Navigate to `http://localhost:5173` to view the application.

### Available Scripts

- `npm run dev` - Start the development server
- `npm run build` - Build the application for production
- `npm run preview` - Preview the production build locally
- `npm run lint` - Run ESLint for code quality checks

## How to Use

1. **Enter a Prompt**: Type or paste your AI prompt in the main input area
2. **Select Attack Types**: Choose which security tests to run:
   - **Jailbreak**: Tests for prompt injection attacks
   - **Hallucination**: Tests for misinformation generation
   - **Advanced**: Additional sophisticated attack vectors
3. **Analyze**: Click the "Analyze Security" button to start testing
4. **Review Results**: Examine the security scores, vulnerability breakdown, and recommendations
5. **Implement Fixes**: Use the provided recommendations to improve your prompt security

## Technology Stack

- **React 18** - Modern React with hooks and functional components
- **TypeScript** - Type-safe JavaScript development
- **Vite** - Fast build tool and development server
- **Tailwind CSS** - Utility-first CSS framework
- **shadcn/ui** - Beautiful and accessible UI components
- **React Query** - Server state management
- **React Router** - Client-side routing

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── ui/             # shadcn/ui components
│   ├── Dashboard.tsx   # Main dashboard component
│   ├── Sidebar.tsx     # Navigation sidebar
│   └── MainContent.tsx # Main content area
├── pages/              # Page components
├── services/           # API services
├── hooks/              # Custom React hooks
└── lib/                # Utility functions
```

## Development

### Code Quality

This project uses:
- ESLint for code linting
- TypeScript for type checking
- Prettier for code formatting (configured in your editor)

### Contributing

1. Create a feature branch from main
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## Deployment

### Using Lovable (Recommended)

1. Visit your [Lovable Project](https://lovable.dev/projects/ed24e900-e810-499a-8af5-bbf71fb1ad07)
2. Click "Share" → "Publish"
3. Your app will be deployed automatically

### Manual Deployment

1. Build the project: `npm run build`
2. Deploy the `dist/` folder to your hosting provider
3. Configure your hosting to serve `index.html` for all routes (SPA routing)

## Custom Domain

To connect a custom domain:
1. Navigate to Project > Settings > Domains in Lovable
2. Click "Connect Domain"
3. Follow the setup instructions

## Support

- [Lovable Documentation](https://docs.lovable.dev/)
- [Discord Community](https://discord.com/channels/1119885301872070706/1280461670979993613)
- [Video Tutorials](https://www.youtube.com/watch?v=9KHLTZaJcR8&list=PLbVHz4urQBZkJiAWdG8HWoJTdgEysigIO)

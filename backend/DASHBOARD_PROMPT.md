# Redact Dashboard Generation Prompt

Create a professional AI prompt security dashboard called "Redact" using **ONLY Tailwind CSS and shadcn/ui components**. This is a serious B2B tool for AI teams to stress-test their prompts before deployment.

## 🔧 **Backend API Endpoints**

The dashboard must integrate with these ALREADY BUILT specific backend endpoints:

### **Core Endpoints**
- `POST /api/v1/attacks/generate` - Generate attack prompts with rate limiting
- `POST /api/v1/attacks/test-resistance` - Test prompt resistance and return analysis
- `GET /api/v1/attacks/stats` - Comprehensive system statistics (Redis features, metrics, rate limiting)

### **Pipeline Endpoints**
- `GET /api/v1/attacks/pipeline/stats` - Real-time pipeline statistics (executor worker, evaluator status)
- `GET /api/v1/attacks/pipeline/verdicts?limit=10` - Recent verdicts with detailed evaluations
- `GET /api/v1/attacks/pipeline/results?limit=10` - Recent attack results from executor worker

### **Job Queue Endpoints**
- `POST /api/v1/attacks/submit-job` - Submit background job for attack generation
- `GET /api/v1/attacks/job/{job_id}` - Get job status and results
- `GET /api/v1/attacks/queue/stats` - Job queue statistics
- `GET /api/v1/attacks/queue/pending?limit=10` - List of pending jobs

### **Health & Monitoring**
- `GET /api/v1/attacks/health` - Health check for attack generation service
- `GET /` - Basic API information

### **Rate Limiting**
- All endpoints implement Redis-based rate limiting
- Attack Generation: 50 requests/hour per user, 100 per IP
- API Calls: 200 requests/hour per user, 500 per IP
- Returns 429 status with detailed rate limit information

### **Real-time Features**
- Redis Streams for attack queue processing
- Redis Pub/Sub for live verdict updates
- Redis Cache for attack and response deduplication
- Comprehensive metrics collection for judges

## 🎨 **Design System & CSS Variables**

Use these exact CSS variables throughout the application:

```css
:root {
  --background: oklch(0.8798 0.0534 91.7893);
  --foreground: oklch(0.4265 0.0310 59.2153);
  --card: oklch(0.8937 0.0395 87.5676);
  --card-foreground: oklch(0.4265 0.0310 59.2153);
  --popover: oklch(0.9378 0.0331 89.8515);
  --popover-foreground: oklch(0.4265 0.0310 59.2153);
  --primary: oklch(0.6657 0.1050 118.9078);
  --primary-foreground: oklch(0.9882 0.0069 88.6415);
  --secondary: oklch(0.8532 0.0631 91.1493);
  --secondary-foreground: oklch(0.4265 0.0310 59.2153);
  --muted: oklch(0.8532 0.0631 91.1493);
  --muted-foreground: oklch(0.5761 0.0259 60.9323);
  --accent: oklch(0.8361 0.0713 90.3269);
  --accent-foreground: oklch(0.4265 0.0310 59.2153);
  --destructive: oklch(0.7136 0.0981 29.9827);
  --destructive-foreground: oklch(0.9790 0.0082 91.4818);
  --border: oklch(0.6918 0.0440 59.8448);
  --input: oklch(0.8361 0.0713 90.3269);
  --ring: oklch(0.7350 0.0564 130.8494);
  --chart-1: oklch(0.7350 0.0564 130.8494);
  --chart-2: oklch(0.6762 0.0567 132.4479);
  --chart-3: oklch(0.8185 0.0332 136.6539);
  --chart-4: oklch(0.5929 0.0464 137.6224);
  --chart-5: oklch(0.5183 0.0390 137.1892);
  --sidebar: oklch(0.8631 0.0645 90.5161);
  --sidebar-foreground: oklch(0.4265 0.0310 59.2153);
  --sidebar-primary: oklch(0.7350 0.0564 130.8494);
  --sidebar-primary-foreground: oklch(0.9882 0.0069 88.6415);
  --sidebar-accent: oklch(0.9225 0.0169 88.0027);
  --sidebar-accent-foreground: oklch(0.4265 0.0310 59.2153);
  --sidebar-border: oklch(0.9073 0.0170 88.0044);
  --sidebar-ring: oklch(0.7350 0.0564 130.8494);
  --font-sans: Merriweather, serif;
  --font-serif: Source Serif 4, serif;
  --font-mono: JetBrains Mono, monospace;
  --radius: 0.425rem;
  --shadow-2xs: 3px 3px 2px 0px hsl(88 22% 35% / 0.07);
  --shadow-xs: 3px 3px 2px 0px hsl(88 22% 35% / 0.07);
  --shadow-sm: 3px 3px 2px 0px hsl(88 22% 35% / 0.15), 3px 1px 2px -1px hsl(88 22% 35% / 0.15);
  --shadow: 3px 3px 2px 0px hsl(88 22% 35% / 0.15), 3px 1px 2px -1px hsl(88 22% 35% / 0.15);
  --shadow-md: 3px 3px 2px 0px hsl(88 22% 35% / 0.15), 3px 2px 4px -1px hsl(88 22% 35% / 0.15);
  --shadow-lg: 3px 3px 2px 0px hsl(88 22% 35% / 0.15), 3px 4px 6px -1px hsl(88 22% 35% / 0.15);
  --shadow-xl: 3px 3px 2px 0px hsl(88 22% 35% / 0.15), 3px 8px 10px -1px hsl(88 22% 35% / 0.15);
  --shadow-2xl: 3px 3px 2px 0px hsl(88 22% 35% / 0.38);
  --tracking-normal: 0em;
  --spacing: 0.25rem;
}

.dark {
  --background: oklch(0.3303 0.0214 88.0737);
  --foreground: oklch(0.9217 0.0235 82.1191);
  --card: oklch(0.3583 0.0165 82.3257);
  --card-foreground: oklch(0.9217 0.0235 82.1191);
  --popover: oklch(0.3583 0.0165 82.3257);
  --popover-foreground: oklch(0.9217 0.0235 82.1191);
  --primary: oklch(0.6762 0.0567 132.4479);
  --primary-foreground: oklch(0.2686 0.0105 61.0213);
  --secondary: oklch(0.4448 0.0239 84.5498);
  --secondary-foreground: oklch(0.9217 0.0235 82.1191);
  --muted: oklch(0.3892 0.0197 82.7084);
  --muted-foreground: oklch(0.7096 0.0171 73.6179);
  --accent: oklch(0.6540 0.0723 90.7629);
  --accent-foreground: oklch(0.2686 0.0105 61.0213);
  --destructive: oklch(0.6287 0.0821 31.2958);
  --destructive-foreground: oklch(0.9357 0.0201 84.5907);
  --border: oklch(0.4448 0.0239 84.5498);
  --input: oklch(0.4448 0.0239 84.5498);
  --ring: oklch(0.6762 0.0567 132.4479);
  --chart-1: oklch(0.6762 0.0567 132.4479);
  --chart-2: oklch(0.7350 0.0564 130.8494);
  --chart-3: oklch(0.5929 0.0464 137.6224);
  --chart-4: oklch(0.6540 0.0723 90.7629);
  --chart-5: oklch(0.5183 0.0390 137.1892);
  --sidebar: oklch(0.3303 0.0214 88.0737);
  --sidebar-foreground: oklch(0.9217 0.0235 82.1191);
  --sidebar-primary: oklch(0.6762 0.0567 132.4479);
  --sidebar-primary-foreground: oklch(0.2686 0.0105 61.0213);
  --sidebar-accent: oklch(0.6540 0.0723 90.7629);
  --sidebar-accent-foreground: oklch(0.2686 0.0105 61.0213);
  --sidebar-border: oklch(0.4448 0.0239 84.5498);
  --sidebar-ring: oklch(0.6762 0.0567 132.4479);
  --font-sans: Merriweather, serif;
  --font-serif: Source Serif 4, serif;
  --font-mono: JetBrains Mono, monospace;
  --radius: 0.425rem;
  --shadow-2xs: 3px 3px 2px 0px hsl(88 22% 35% / 0.07);
  --shadow-xs: 3px 3px 2px 0px hsl(88 22% 35% / 0.07);
  --shadow-sm: 3px 3px 2px 0px hsl(88 22% 35% / 0.15), 3px 1px 2px -1px hsl(88 22% 35% / 0.15);
  --shadow: 3px 3px 2px 0px hsl(88 22% 35% / 0.15), 3px 1px 2px -1px hsl(88 22% 35% / 0.15);
  --shadow-md: 3px 3px 2px 0px hsl(88 22% 35% / 0.15), 3px 2px 4px -1px hsl(88 22% 35% / 0.15);
  --shadow-lg: 3px 3px 2px 0px hsl(88 22% 35% / 0.15), 3px 4px 6px -1px hsl(88 22% 35% / 0.15);
  --shadow-xl: 3px 3px 2px 0px hsl(88 22% 35% / 0.15), 3px 8px 10px -1px hsl(88 22% 35% / 0.15);
  --shadow-2xl: 3px 3px 2px 0px hsl(88 22% 35% / 0.38);
}
```

## 🚀 **CRITICAL: shadcn/ui Components ONLY**

**ABSOLUTE REQUIREMENT**: Use **ONLY** shadcn/ui components. Do NOT create custom components or use other UI libraries. This is non-negotiable.

### **Install Required Components**
```bash
npx shadcn@latest add card button input textarea table badge progress alert tabs select checkbox switch label separator scroll-area sheet navigation-menu
```

### **Essential shadcn/ui Components**
- `<Card>`, `<CardContent>`, `<CardHeader>`, `<CardTitle>`, `<CardDescription>` - For all content areas
- `<Button>` - For all actions (use variants: default, destructive, outline, secondary, ghost)
- `<Input>`, `<Textarea>` - For all text inputs
- `<Table>`, `<TableBody>`, `<TableCell>`, `<TableHead>`, `<TableHeader>`, `<TableRow>` - For data display
- `<Badge>` - For status indicators and risk levels
- `<Progress>` - For evaluation progress
- `<Alert>`, `<AlertDescription>`, `<AlertTitle>` - For notifications
- `<Tabs>`, `<TabsContent>`, `<TabsList>`, `<TabsTrigger>` - For organizing views
- `<Select>`, `<SelectContent>`, `<SelectItem>`, `<SelectTrigger>`, `<SelectValue>` - For dropdowns
- `<Checkbox>`, `<Switch>` - For toggles and selections
- `<Label>` - For form labels
- `<Separator>` - For visual dividers
- `<ScrollArea>` - For scrollable content
- `<Sheet>` - For mobile sidebar
- `<NavigationMenu>` - For main navigation

## 🏗️ **Layout Structure**

### **Main Layout**
- Use `<Sheet>` for mobile sidebar
- Use `<NavigationMenu>` for main navigation
- Use `<Card>` for all content areas
- Use `<Separator>` for visual dividers

### **Key Pages**
- **Dashboard**: Stats cards, recent activity, quick actions
- **Attack Generator**: Prompt input, attack configuration, generate button
- **Live Stream**: Real-time attack results and verdicts
- **Reports**: Test summaries, risk analysis, download options

### **Essential Components**
- `<Card>` with `<CardHeader>` and `<CardContent>` for all sections
- `<Table>` for displaying attack data and results
- `<Badge>` for status indicators (safe, high risk, medium risk)
- `<Button>` for all actions (primary, destructive, outline variants)
- `<Progress>` for evaluation progress
- `<Alert>` for notifications and warnings

### **Component Variants**
```tsx
// Use shadcn/ui variants consistently
<Badge variant="default">Safe</Badge>
<Badge variant="destructive">High Risk</Badge>
<Badge variant="secondary">Medium Risk</Badge>
<Badge variant="outline">Info</Badge>

<Button variant="default">Primary Action</Button>
<Button variant="destructive">Delete</Button>
<Button variant="outline">Secondary</Button>
<Button variant="ghost">Subtle</Button>
```

## 🎯 **Layout Examples (Based on Images)**

### **Dashboard Layout**
- **Left Sidebar**: Navigation with "Redact Security" branding, navigation items with icons
- **Main Content**: Header with "Security Dashboard" title and timestamp
- **System Status Banner**: Green `<Alert>` showing "System Status: Operational"
- **Stats Cards**: Four `<Card>` components showing:
  - Total Attacks: "15,463" with description
  - Successful Attacks: "247" with warning icon
  - Avg Resistance: "84%" with "High" badge
  - Active Tests: "12" with description
- **Recent Activity Table**: `<Table>` with columns: Timestamp, Attack Type, Target, Result, Risk Level
- **Quick Actions**: `<Card>` with action buttons

### **Component Usage Examples**
```tsx
// Stats Card
<Card>
  <CardHeader>
    <CardTitle>Total Attacks</CardTitle>
  </CardHeader>
  <CardContent>
    <div className="text-2xl font-bold">15,463</div>
    <p className="text-sm text-muted-foreground">All time attack attempts</p>
  </CardContent>
</Card>

// Status Badge
<Badge variant="destructive">High Risk</Badge>
<Badge variant="default">Blocked</Badge>

// System Status
<Alert>
  <AlertTitle>System Status: Operational</AlertTitle>
  <AlertDescription>All attack generation and monitoring services are running normally.</AlertDescription>
</Alert>
```

## 🚫 **What NOT to Do**
- ❌ Don't create custom components
- ❌ Don't use other UI libraries
- ❌ Don't add chat-like interfaces
- ❌ Don't use playful animations
- ❌ Don't hide technical details
- ❌ Don't create custom color schemes

## ✅ **What TO Do**
- ✅ Use ONLY shadcn/ui components
- ✅ Follow the provided CSS variables exactly
- ✅ Use the specified fonts (Merriweather, Source Serif 4, JetBrains Mono)
- ✅ Create a professional, data-driven interface
- ✅ Keep it simple and relevant - no clunky interfaces
- ✅ Make it feel like a security testing tool

This dashboard should feel like a professional pen-testing station for AI prompts, not a chat application. 
# Video Processing Comparison Frontend

This is a React/TypeScript application for comparing video processing capabilities of different cloud providers (AWS Rekognition, Azure AI Vision) and a local YOLO model.

## Features

- Upload videos for processing
- Compare processing time, detection counts, and costs between providers
- View detailed metrics and performance data
- Clean, modern UI built with Ant Design

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```
3. Create a `.env` file with your API URL:
   ```
   VITE_API_URL=http://localhost:8080
   ```

### Development

Run the development server:

```bash
npm run dev
```

### Building for Production

```bash
npm run build
```

## Technologies Used

- React
- TypeScript
- Vite
- Ant Design
- Axios
- Recharts 
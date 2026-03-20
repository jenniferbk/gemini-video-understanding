# React 19 + TypeScript Development Patterns

Best practices and patterns for building React 19 applications with TypeScript, including component structure, hooks, error handling, and state management.

## When to use this skill

- Creating React components in `src/renderer/`
- Implementing custom hooks
- Managing component state and side effects
- Working with TypeScript types for React
- Error handling in React applications
- Component testing patterns

## Project Stack

- **React 19** with TypeScript
- **Material UI v7** (MUI) for components
- **TanStack Query v5** for server state
- **TanStack Router** with file-based routing
- **CSS Modules** for component styling

## Component Structure

### File Organization

```
src/renderer/components/
├── VideoUpload/
│   ├── VideoUpload.tsx          # Main component
│   ├── VideoUpload.module.css   # Scoped styles
│   └── VideoUpload.test.tsx     # Tests
├── shared/
│   ├── Button.tsx               # Reusable components
│   ├── Input.tsx
│   └── Modal.tsx
```

### Component Template

```typescript
// src/renderer/components/VideoUpload/VideoUpload.tsx
import React, { useState, useCallback } from 'react';
import styles from './VideoUpload.module.css';

// Props interface
interface VideoUploadProps {
  onVideoSelected: (file: File) => void;
  accept?: string;
  maxSizeMB?: number;
}

// Component
export const VideoUpload: React.FC<VideoUploadProps> = ({
  onVideoSelected,
  accept = '.mp4,.mov,.avi',
  maxSizeMB = 500
}) => {
  // State
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Handlers
  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    setError(null);

    const file = e.dataTransfer.files[0];
    if (file && isValidVideoFile(file, maxSizeMB)) {
      onVideoSelected(file);
    } else {
      setError(`Invalid file. Max size: ${maxSizeMB}MB`);
    }
  }, [onVideoSelected, maxSizeMB]);

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragActive(false);
  }, []);

  // Render
  return (
    <div
      className={`${styles.dropzone} ${dragActive ? styles.active : ''}`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      <p className={styles.text}>
        Drag video here or click to browse
      </p>
      {error && <p className={styles.error}>{error}</p>}
    </div>
  );
};

// Helper functions
function isValidVideoFile(file: File, maxSizeMB: number): boolean {
  const validExtensions = ['.mp4', '.mov', '.avi'];
  const maxBytes = maxSizeMB * 1024 * 1024;

  const hasValidExtension = validExtensions.some(ext =>
    file.name.toLowerCase().endsWith(ext)
  );

  return hasValidExtension && file.size <= maxBytes;
}
```

**Key Patterns:**
- Export named components (not default)
- Define props interface above component
- Use `React.FC<Props>` for type safety
- Use CSS Modules for scoped styling
- Keep helper functions below component or in separate utils file
- Use `useCallback` for event handlers to prevent re-renders

## Custom Hooks

### Hook Template

```typescript
// src/renderer/hooks/useTranscription.ts
import { useState, useEffect, useCallback, useRef } from 'react';

export interface TranscriptionConfig {
  videoPath: string;
  prompt: string;
  consensusRuns: number;
}

export interface ProgressUpdate {
  chunk: number;
  total: number;
  percent: number;
  status: string;
}

export function useTranscription() {
  // State
  const [progress, setProgress] = useState<ProgressUpdate | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Refs for cleanup
  const cleanupRef = useRef<(() => void) | null>(null);

  // Subscribe to progress events
  useEffect(() => {
    const cleanup = window.electronAPI.onProgress((update: ProgressUpdate) => {
      setProgress(update);
    });

    cleanupRef.current = cleanup;

    return () => {
      cleanup();
    };
  }, []);

  // Start transcription
  const startTranscription = useCallback(async (config: TranscriptionConfig) => {
    try {
      setError(null);
      setIsRunning(true);

      const result = await window.electronAPI.startTranscription(config);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      setIsRunning(false);
      throw err;
    }
  }, []);

  // Cancel transcription
  const cancelTranscription = useCallback(async () => {
    try {
      await window.electronAPI.cancelTranscription();
      setIsRunning(false);
      setProgress(null);
    } catch (err) {
      console.error('Failed to cancel:', err);
    }
  }, []);

  return {
    progress,
    isRunning,
    error,
    startTranscription,
    cancelTranscription
  };
}
```

**Hook Best Practices:**
- Prefix with `use`
- Return object with named properties
- Use `useCallback` for returned functions
- Handle cleanup in `useEffect`
- Manage loading/error states
- Use refs for cleanup functions

## Error Handling

### Error Boundary

```typescript
// src/renderer/components/ErrorBoundary.tsx
import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error:', error, errorInfo);
    // Optionally log to external service
  }

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div style={{ padding: '20px' }}>
          <h2>Something went wrong</h2>
          <details>
            <summary>Error details</summary>
            <pre>{this.state.error?.message}</pre>
          </details>
        </div>
      );
    }

    return this.props.children;
  }
}

// Usage
<ErrorBoundary>
  <App />
</ErrorBoundary>
```

### Try-Catch Pattern

```typescript
// Component with error handling
const ConfigScreen: React.FC = () => {
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (config: Config) => {
    try {
      setError(null);
      setLoading(true);

      await window.electronAPI.saveConfig(config);

      // Success handling
      navigate('/progress');
    } catch (err) {
      // Type-safe error handling
      if (err instanceof ApiKeyError) {
        setError('Invalid API key. Please check your settings.');
      } else if (err instanceof NetworkError) {
        setError('Network error. Check your internet connection.');
      } else {
        setError('An unexpected error occurred. Please try again.');
        console.error('Config save error:', err);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {error && <Alert severity="error">{error}</Alert>}
      {/* ... rest of component */}
    </div>
  );
};
```

### Custom Error Classes

```typescript
// src/renderer/types/errors.ts
export class ApiKeyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ApiKeyError';
  }
}

export class NetworkError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'NetworkError';
  }
}

export class ValidationError extends Error {
  constructor(
    message: string,
    public field: string
  ) {
    super(message);
    this.name = 'ValidationError';
  }
}
```

## State Management

### Local State (useState)

```typescript
// Simple component state
const [value, setValue] = useState<string>('');
const [isOpen, setIsOpen] = useState(false);
const [items, setItems] = useState<Item[]>([]);

// Complex state with interface
interface FormState {
  videoPath: string;
  prompt: string;
  settings: Settings;
}

const [formState, setFormState] = useState<FormState>({
  videoPath: '',
  prompt: '',
  settings: defaultSettings
});

// Update complex state
const updateField = (field: keyof FormState, value: any) => {
  setFormState(prev => ({
    ...prev,
    [field]: value
  }));
};
```

### Context for Shared State

```typescript
// src/renderer/contexts/SettingsContext.tsx
import React, { createContext, useContext, useState, useCallback } from 'react';

interface Settings {
  theme: 'light' | 'dark';
  apiKey: string | null;
  outputPath: string;
}

interface SettingsContextValue {
  settings: Settings;
  updateSetting: <K extends keyof Settings>(key: K, value: Settings[K]) => void;
  loading: boolean;
}

const SettingsContext = createContext<SettingsContextValue | undefined>(undefined);

export const SettingsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [settings, setSettings] = useState<Settings>({
    theme: 'light',
    apiKey: null,
    outputPath: '~/Documents/VideoTranscripts'
  });
  const [loading, setLoading] = useState(false);

  const updateSetting = useCallback(<K extends keyof Settings>(
    key: K,
    value: Settings[K]
  ) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));

    // Persist to electron
    window.electronAPI.setSetting(key, String(value));
  }, []);

  return (
    <SettingsContext.Provider value={{ settings, updateSetting, loading }}>
      {children}
    </SettingsContext.Provider>
  );
};

// Custom hook for consuming context
export function useSettings() {
  const context = useContext(SettingsContext);
  if (!context) {
    throw new Error('useSettings must be used within SettingsProvider');
  }
  return context;
}
```

## Material UI Patterns

### Theme Configuration

```typescript
// src/renderer/theme.ts
import { createTheme } from '@mui/material/styles';

export const theme = createTheme({
  palette: {
    primary: {
      main: '#3b82f6',
      light: '#60a5fa',
      dark: '#2563eb',
    },
    error: {
      main: '#ef4444',
    },
    success: {
      main: '#10b981',
    },
  },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      'sans-serif',
    ].join(','),
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none', // Disable uppercase
        },
      },
    },
  },
});

// App.tsx
import { ThemeProvider } from '@mui/material/styles';
import { theme } from './theme';

function App() {
  return (
    <ThemeProvider theme={theme}>
      {/* app content */}
    </ThemeProvider>
  );
}
```

### MUI Component Usage

```typescript
import { Button, TextField, CircularProgress } from '@mui/material';

function ConfigForm() {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);

  return (
    <Box sx={{ p: 3 }}>
      <TextField
        label="Prompt Name"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        fullWidth
        margin="normal"
      />

      <Button
        variant="contained"
        onClick={handleSubmit}
        disabled={loading}
        startIcon={loading ? <CircularProgress size={20} /> : null}
      >
        {loading ? 'Processing...' : 'Start Transcription'}
      </Button>
    </Box>
  );
}
```

## TypeScript Patterns

### Strict Type Definitions

```typescript
// src/renderer/types/transcription.ts

// Use exact types, not 'any'
export interface TranscriptionJob {
  id: number;
  videoPath: string;
  promptName: string;
  config: TranscriptionConfig;
  status: JobStatus; // Not 'string'
  createdAt: Date;
}

// Use union types for fixed sets
export type JobStatus =
  | 'queued'
  | 'processing'
  | 'complete'
  | 'failed'
  | 'cancelled';

// Use generics for reusable types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// Usage
const response: ApiResponse<TranscriptionJob> = await api.getJob(id);
```

### Type Guards

```typescript
// src/renderer/utils/typeGuards.ts

export function isProgressUpdate(data: unknown): data is ProgressUpdate {
  return (
    typeof data === 'object' &&
    data !== null &&
    'type' in data &&
    data.type === 'progress'
  );
}

export function isError(error: unknown): error is Error {
  return error instanceof Error;
}

// Usage
if (isProgressUpdate(message)) {
  // TypeScript knows message is ProgressUpdate
  console.log(message.percent);
}
```

## Performance Optimization

### Memoization

```typescript
import { useMemo, useCallback } from 'react';

function ExpensiveComponent({ items, filter }: Props) {
  // Memoize expensive calculation
  const filteredItems = useMemo(() => {
    return items.filter(item => item.name.includes(filter));
  }, [items, filter]);

  // Memoize callback
  const handleClick = useCallback((id: number) => {
    console.log('Clicked:', id);
  }, []);

  return (
    <div>
      {filteredItems.map(item => (
        <Item key={item.id} onClick={handleClick} />
      ))}
    </div>
  );
}
```

### React.memo

```typescript
// Prevent unnecessary re-renders
export const ListItem = React.memo<ListItemProps>(({ item, onClick }) => {
  return (
    <div onClick={() => onClick(item.id)}>
      {item.name}
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison
  return prevProps.item.id === nextProps.item.id;
});
```

## Testing Patterns

```typescript
// VideoUpload.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { VideoUpload } from './VideoUpload';

describe('VideoUpload', () => {
  it('calls onVideoSelected when valid file dropped', () => {
    const mockOnSelect = jest.fn();
    render(<VideoUpload onVideoSelected={mockOnSelect} />);

    const file = new File(['video'], 'test.mp4', { type: 'video/mp4' });
    const dropzone = screen.getByText(/drag video here/i);

    fireEvent.drop(dropzone, {
      dataTransfer: { files: [file] }
    });

    expect(mockOnSelect).toHaveBeenCalledWith(file);
  });

  it('shows error for oversized file', () => {
    render(<VideoUpload maxSizeMB={1} onVideoSelected={() => {}} />);

    const largeFile = new File(
      [new Array Blob(2 * 1024 * 1024)],
      'large.mp4'
    );

    const dropzone = screen.getByText(/drag video here/i);
    fireEvent.drop(dropzone, {
      dataTransfer: { files: [largeFile] }
    });

    expect(screen.getByText(/invalid file/i)).toBeInTheDocument();
  });
});
```

## Common Patterns

### Loading States

```typescript
const [data, setData] = useState<Data | null>(null);
const [loading, setLoading] = useState(true);
const [error, setError] = useState<Error | null>(null);

useEffect(() => {
  async function fetchData() {
    try {
      setLoading(true);
      const result = await api.getData();
      setData(result);
    } catch (err) {
      setError(err as Error);
    } finally {
      setLoading(false);
    }
  }

  fetchData();
}, []);

if (loading) return <CircularProgress />;
if (error) return <Alert severity="error">{error.message}</Alert>;
if (!data) return null;

return <div>{/* render data */}</div>;
```

### Form Handling

```typescript
const [formValues, setFormValues] = useState({
  prompt: '',
  chunkMinutes: 2,
  consensusRuns: 3
});

const handleChange = (field: string) => (
  e: React.ChangeEvent<HTMLInputElement>
) => {
  setFormValues(prev => ({
    ...prev,
    [field]: e.target.value
  }));
};

<TextField
  value={formValues.prompt}
  onChange={handleChange('prompt')}
/>
```

## Additional Resources

- React 19 Docs: https://react.dev
- MUI Documentation: https://mui.com
- TypeScript Handbook: https://www.typescriptlang.org/docs/
- See `PROJECT_KNOWLEDGE.md` for application architecture

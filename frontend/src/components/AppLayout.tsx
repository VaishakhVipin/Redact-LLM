import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { TestSupabase } from './TestSupabase';

export const AppLayout = () => {
  return (
    <div className="flex h-screen bg-background font-sans">
      <TestSupabase />
      <Sidebar />
      <main className="flex-1 overflow-y-auto pl-64">
        <div className="min-h-full flex items-center justify-center p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
};

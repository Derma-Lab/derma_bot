// src/app/page.tsx

import ChatWindow from '../components/ChatWindow';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gray-100 dotted-background">
      <ChatWindow />
    </div>
  );
}

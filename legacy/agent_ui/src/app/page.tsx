// src/app/page.tsx

import ChatWindow from '../components/ChatWindow';

export default function HomePage() {
  return (
    <div
      className="min-h-screen"
      style={{
        backgroundImage: 'url(/isometric.jpg)',
        backgroundSize: 'cover',
        backgroundAttachment: 'fixed',
      }}
    >
      <ChatWindow />
    </div>
  );
}

export interface Message {
  content: string;
  sender: 'user' | 'agent';
  type: 'message' | 'card';
}

export interface Card {
  id: number;
  content: string;
  position: { x: number; y: number };
  zIndex: number;
} 
// src/components/ChatWindow.tsx
'use client';

import React, { useState, useEffect, FormEvent, useRef } from 'react';
import { DraggableCore } from 'react-draggable';
import { processInput } from '../utils/api';
import Card from './Card';
import { Message, Card as CardType } from '../types/types';

const ChatWindow: React.FC = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [cards, setCards] = useState<CardType[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [zIndexCounter, setZIndexCounter] = useState(1);
  const chatRef = useRef<HTMLDivElement>(null);
  const nodeRef = useRef(null);
  const [position, setPosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    setMessages([
      {
        content: "👩‍⚕️ Nurse: Hello, I'll be gathering some information about your skin condition.\n\nPlease describe your main skin concerns.",
        sender: 'agent',
        type: 'message',
      },
    ]);
  }, []);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    setMessages(prev => [...prev, { content: input, sender: 'user', type: 'message' }]);
    const currentInput = input;
    setInput('');
    setIsTyping(true);

    try {
      const response = await processInput(currentInput);
      
      response.messages.forEach(msg => {
        if (msg.type === 'card') {
          const newCard: CardType = {
            id: Date.now(),
            content: msg.content,
            position: {
              x: Math.random() * (window.innerWidth - 300),
              y: Math.random() * (window.innerHeight - 200),
            },
            zIndex: zIndexCounter,
          };
          setCards(prev => [...prev, newCard]);
          setZIndexCounter(prev => prev + 1);
        } else {
          setMessages(prev => [...prev, msg]);
        }
      });
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setIsTyping(false);
    }
  };

  const handleCardPosition = (id: number, newPosition: { x: number; y: number }) => {
    setCards(prev => 
      prev.map(card => card.id === id ? { ...card, position: newPosition } : card)
    );
  };

  const handleBringToFront = (id: number) => {
    setZIndexCounter(prev => prev + 1);
    setCards(prev =>
      prev.map(card => card.id === id ? { ...card, zIndex: zIndexCounter } : card)
    );
  };

  const handleDrag = (e: any, data: any) => {
    setPosition({
      x: position.x + data.deltaX,
      y: position.y + data.deltaY,
    });
  };

  const addNewCard = () => {
    const newCard: CardType = {
      id: Date.now(),
      content: 'New card',
      position: {
        x: Math.random() * (window.innerWidth - 300),
        y: Math.random() * (window.innerHeight - 200),
      },
      zIndex: zIndexCounter,
    };
    setCards(prev => [...prev, newCard]);
    setZIndexCounter(prev => prev + 1);
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (nodeRef.current) {
      setPosition(prev => ({
        x: prev.x - e.deltaX,
        y: prev.y - e.deltaY,
      }));
    }
  };

  return (
    <>
      <DraggableCore nodeRef={nodeRef} handle=".handle" onDrag={handleDrag}>
        <div
          ref={nodeRef}
          onWheel={handleWheel}
          style={{
            transform: `translate(${position.x}px, ${position.y}px)`,
            position: 'fixed',
            bottom: '20px',
            right: '20px',
          }}
          className="w-[320px] h-[500px] bg-white rounded-lg shadow-lg flex flex-col"
        >
          <div className="handle bg-blue-500 text-white p-3 cursor-move flex justify-between items-center">
            <span>Chat</span>
            <button
              type="button"
              onClick={addNewCard}
              className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600"
            >
              Add Card
            </button>
          </div>
          <div className="flex-1 p-3 overflow-y-auto space-y-2">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`${msg.sender === 'user' ? 'text-right' : 'text-left'}`}
              >
                <span
                  className={`inline-block px-3 py-2 rounded-lg ${
                    msg.sender === 'user'
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-200 text-gray-800'
                  } whitespace-pre-wrap`}
                >
                  {msg.content}
                </span>
              </div>
            ))}
            {isTyping && (
              <div className="text-left">
                <span className="inline-block px-3 py-2 rounded-lg bg-gray-200">
                  Typing...
                </span>
              </div>
            )}
          </div>
          <form onSubmit={handleSubmit} className="p-3 border-t">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="w-full px-3 py-2 border rounded focus:outline-none"
              placeholder="Type a message..."
            />
          </form>
        </div>
      </DraggableCore>

      {cards.map((card) => (
        <Card
          key={card.id}
          {...card}
          onPositionChange={handleCardPosition}
          onBringToFront={handleBringToFront}
        />
      ))}
    </>
  );
};

export default ChatWindow;

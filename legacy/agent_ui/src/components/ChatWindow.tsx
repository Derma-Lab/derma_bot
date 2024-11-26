// src/components/ChatWindow.tsx
'use client';

import React, { useState, useEffect, FormEvent, useRef } from 'react';
import { DraggableCore } from 'react-draggable';
import { processInput, Message } from '../utils/api';
import AgentCard from './AgentCard';

interface AgentCardData {
  id: number;
  content: string;
  position: { x: number; y: number };
  zIndex: number;
}

const ChatWindow: React.FC = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [agentCards, setAgentCards] = useState<AgentCardData[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const nodeRef = useRef(null);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [zIndexCounter, setZIndexCounter] = useState(1);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 320, height: 500 }); // Default dimensions
  const [isResizing, setIsResizing] = useState(false);

  // Handle window resize
  useEffect(() => {
    const maxWidth = Math.min(window.innerWidth * 0.8, 800); // Max 80% of screen width or 800px
    const maxHeight = Math.min(window.innerHeight * 0.8, 800); // Max 80% of screen height or 800px
    setDimensions(prev => ({
      width: Math.min(prev.width, maxWidth),
      height: Math.min(prev.height, maxHeight)
    }));
  }, []);

  const handleProcessInput = async (input: string) => {
    try {
      setIsTyping(true);

      const response = await processInput(input);

      // Update messages
      setMessages((prev) => [...prev, ...response.messages.filter(msg => msg.type === 'message')]);

      // Handle agent cards
      response.messages.forEach((msg) => {
        if (msg.type === 'card') {
          const id = Date.now() + Math.random(); // Ensure unique ID
          const position = {
            x: Math.random() * (window.innerWidth - 400),
            y: Math.random() * (window.innerHeight - 200),
          };

          const newCard: AgentCardData = {
            id,
            content: msg.content,
            position,
            zIndex: zIndexCounter,
          };

          setAgentCards((prev) => [...prev, newCard]);
          setZIndexCounter((prev) => prev + 1);
        }
      });

      setIsTyping(false);

      if (response.endOfConversation) {
        // Handle end of conversation if needed
      }
    } catch (error) {
      console.error('Error processing input:', error);
      setIsTyping(false);
    }
  };

  // Initialize conversation on mount
  useEffect(() => {
    // Start the conversation by adding the initial message
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

    // Add user's message
    const userMessage: Message = { content: input, sender: 'user', type: 'message' };
    setMessages((prev) => [...prev, userMessage]);

    // Clear input
    setInput('');

    // Process the input and get the next response
    await handleProcessInput(input);
  };

  const handlePositionChange = (id: number, newPosition: { x: number; y: number }) => {
    setAgentCards((prevCards) =>
      prevCards.map((card) =>
        card.id === id ? { ...card, position: newPosition } : card
      )
    );
  };

  const handleBringToFront = (id: number) => {
    setZIndexCounter((prev) => prev + 1);
    setAgentCards((prevCards) =>
      prevCards.map((card) =>
        card.id === id ? { ...card, zIndex: zIndexCounter } : card
      )
    );
  };

  const handleDrag = (e: any, data: any) => {
    setPosition({
      x: position.x + data.deltaX,
      y: position.y + data.deltaY,
    });
  };

  const handleResize = (e: MouseEvent) => {
    const maxWidth = window.innerWidth * 0.8;
    const maxHeight = window.innerHeight * 0.8;
    const minWidth = 320;
    const minHeight = 400;

    if (nodeRef.current) {
      const element = nodeRef.current as HTMLElement;
      const newWidth = Math.min(Math.max(e.clientX - element.getBoundingClientRect().left, minWidth), maxWidth);
      const newHeight = Math.min(Math.max(e.clientY - element.getBoundingClientRect().top, minHeight), maxHeight);
      
      setDimensions({ width: newWidth, height: newHeight });
    }
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isResizing) {
        handleResize(e);
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing]);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle scroll button visibility
  const handleScroll = () => {
    if (chatContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = chatContainerRef.current;
      setShowScrollButton(scrollHeight - scrollTop - clientHeight > 100);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <>
      <DraggableCore nodeRef={nodeRef} handle=".handle" onDrag={handleDrag}>
        <div
          ref={nodeRef}
          style={{
            transform: `translate(${position.x}px, ${position.y}px)`,
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            width: `${dimensions.width}px`,
            height: `${dimensions.height}px`,
            minWidth: '320px',
            minHeight: '400px',
            maxWidth: '80vw',
            maxHeight: '80vh',
            resize: 'both',
            overflow: 'hidden'
          }}
          className="bg-white rounded-lg shadow-lg flex flex-col"
        >
          <div className="bg-blue-500 text-white p-3 handle cursor-move">
            Chat
          </div>
          <div 
            ref={chatContainerRef}
            onScroll={handleScroll}
            className="flex-1 p-3 overflow-y-auto space-y-2 relative"
            style={{
              maxHeight: `calc(${dimensions.height}px - 110px)`, // Account for header and input
              overflowY: 'auto',
              scrollBehavior: 'smooth'
            }}
          >
            {showScrollButton && (
              <button
                onClick={scrollToBottom}
                className="fixed bottom-24 right-24 bg-blue-500 text-white rounded-full p-2 shadow-lg hover:bg-blue-600 transition-colors"
                style={{ zIndex: 1000 }}
              >
                ↓
              </button>
            )}
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`${
                  msg.sender === 'user' ? 'text-right' : 'text-left'
                }`}
              >
                <span
                  className={`inline-block px-3 py-2 rounded-lg ${
                    msg.sender === 'user'
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-200 text-gray-800'
                  } whitespace-pre-wrap break-words max-w-[90%]`}
                >
                  {msg.content}
                </span>
              </div>
            ))}
            {isTyping && (
              <div className="text-left">
                <span className="inline-block px-3 py-2 rounded-lg bg-gray-200 text-gray-800">
                  Typing...
                </span>
              </div>
            )}
            <div ref={messagesEndRef} />
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
          <div
            className="absolute bottom-0 right-0 w-4 h-4 cursor-se-resize"
            onMouseDown={() => setIsResizing(true)}
          />
        </div>
      </DraggableCore>

      {/* Render AgentCards */}
      {agentCards.map((card) => (
        <AgentCard
          key={card.id}
          id={card.id}
          content={card.content}
          position={card.position}
          zIndex={card.zIndex}
          onPositionChange={handlePositionChange}
          onBringToFront={handleBringToFront}
        />
      ))}
    </>
  );
};

export default ChatWindow;

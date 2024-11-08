// src/components/AgentCard.tsx
'use client';

import React, { useState, useRef } from 'react';
import { DraggableCore } from 'react-draggable';

interface AgentCardProps {
  id: number;
  content: string;
  position: { x: number; y: number };
  zIndex: number;
  onPositionChange: (id: number, position: { x: number; y: number }) => void;
  onBringToFront: (id: number) => void;
}

const AgentCard: React.FC<AgentCardProps> = ({
  id,
  content,
  position,
  zIndex,
  onPositionChange,
  onBringToFront,
}) => {
  const nodeRef = useRef(null);

  const handleDrag = (e: any, data: any) => {
    onPositionChange(id, {
      x: data.x,
      y: data.y,
    });
  };

  const handleMouseDown = () => {
    onBringToFront(id);
  };

  return (
    <DraggableCore nodeRef={nodeRef} onDrag={handleDrag}>
      <div
        ref={nodeRef}
        style={{
          transform: `translate(${position.x}px, ${position.y}px)`,
          position: 'absolute',
          zIndex: zIndex,
          width: '400px',
        }}
        className="p-4 bg-white rounded-lg shadow-lg cursor-move"
        onMouseDown={handleMouseDown}
      >
        <div className="text-gray-800 whitespace-pre-wrap">
          {content}
        </div>
      </div>
    </DraggableCore>
  );
};

export default AgentCard;

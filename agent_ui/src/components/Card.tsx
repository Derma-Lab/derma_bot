'use client';

import React, { useRef, useState } from 'react';
import { DraggableCore, DraggableData } from 'react-draggable';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faEdit } from '@fortawesome/free-solid-svg-icons';

interface CardProps {
  id: number;
  content: string;
  position: { x: number; y: number };
  zIndex: number;
  onPositionChange: (id: number, position: { x: number; y: number }) => void;
  onBringToFront: (id: number) => void;
}

const Card: React.FC<CardProps> = ({
  id,
  content,
  position,
  zIndex,
  onPositionChange,
  onBringToFront,
}) => {
  const nodeRef = useRef<HTMLDivElement>(null);
  const initialPosition = useRef({ x: 0, y: 0 });
  const [isEditing, setIsEditing] = useState(false);
  const [editableContent, setEditableContent] = useState(content);

  const handleStart = (e: MouseEvent, data: DraggableData) => {
    if (nodeRef.current) {
      const rect = nodeRef.current.getBoundingClientRect();
      initialPosition.current = {
        x: data.x - rect.left,
        y: data.y - rect.top,
      };
    }
  };

  const handleDrag = (e: MouseEvent, data: DraggableData) => {
    onPositionChange(id, {
      x: data.x - initialPosition.current.x,
      y: data.y - initialPosition.current.y,
    });
  };

  const handleEditClick = () => {
    setIsEditing(true);
  };

  const handleBlur = () => {
    setIsEditing(false);
  };

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setEditableContent(e.target.value);
  };

  return (
    <DraggableCore
      nodeRef={nodeRef}
      onStart={handleStart}
      onDrag={handleDrag}
    >
      <div
        ref={nodeRef}
        style={{
          transform: `translate(${position.x}px, ${position.y}px)`,
          position: 'absolute',
          zIndex: zIndex,
          width: '300px',
        }}
        className="p-4 bg-white rounded-lg shadow-lg cursor-move relative"
        onMouseDown={() => onBringToFront(id)}
      >
        <FontAwesomeIcon
          icon={faEdit}
          className="absolute top-2 right-2 text-gray-500 cursor-pointer hover:text-gray-700"
          onClick={handleEditClick}
        />
        {isEditing ? (
          <textarea
            value={editableContent}
            onChange={handleChange}
            onBlur={handleBlur}
            className="w-full h-full p-2 border rounded focus:outline-none"
            autoFocus
          />
        ) : (
          <div className="text-gray-800 whitespace-pre-wrap">{editableContent}</div>
        )}
      </div>
    </DraggableCore>
  );
};

export default Card; 
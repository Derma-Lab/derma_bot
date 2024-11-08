// src/utils/api.ts

export interface Message {
  content: string;
  sender: 'user' | 'agent';
  type?: 'message' | 'card';
}

export interface ProcessInputResponse {
  messages: Message[];
  state: any;
  endOfConversation?: boolean;
}

export const processInput = async (
  input: string
): Promise<ProcessInputResponse> => {
  const response = await fetch('http://localhost:8000/process_input', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      input: input
    })
  });

  const data = await response.json();
  return data;
};

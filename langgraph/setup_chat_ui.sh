#!/bin/bash

echo "Setting up LangGraph Agent Chat UI..."

# Create chat-ui directory
mkdir -p chat-ui
cd chat-ui

# Use create-agent-chat-app
npx create-agent-chat-app@latest . --template nextjs

# Create .env.local for chat UI configuration
cat > .env.local << EOF
NEXT_PUBLIC_API_URL=http://localhost:2024
NEXT_PUBLIC_ASSISTANT_ID=supervisor
LANGCHAIN_API_KEY=your_langsmith_key_here
EOF

echo "Chat UI setup complete!"
echo "1. Make sure your LangGraph server is running on port 2024"
echo "2. Navigate to chat-ui directory: cd chat-ui"
echo "3. Install dependencies: npm install"
echo "4. Start the chat UI: npm run dev"
echo "5. Open http://localhost:3000 in your browser"
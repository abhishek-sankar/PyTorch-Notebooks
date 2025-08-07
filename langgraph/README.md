# LangGraph Supervisor Demo

A simple LangGraph project using LangGraph Supervisor with two specialized agents:
- **Weather Agent**: Provides weather information for different locations
- **Calculator Agent**: Performs mathematical calculations

## Project Structure

```
langgraph/
├── agents.py              # Weather and Calculator agent implementations
├── supervisor_app.py      # Main supervisor setup and local testing
├── server.py              # FastAPI server for chat UI integration
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (add your API keys)
├── setup_chat_ui.sh       # Script to set up the chat UI
└── README.md              # This file
```

## Setup Instructions

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Edit `.env` file and add your API keys:

```bash
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=langgraph-supervisor-demo
```

### 3. Test Locally (Optional)

Test the supervisor without the chat UI:

```bash
python supervisor_app.py
```

Try these commands:
- "What's the weather in New York?"
- "Calculate 15 + 25"
- "What's 10 * 8?"
- "Weather in London?"

### 4. Start the LangGraph Server

```bash
python server.py
```

This starts the server on `http://localhost:2024`

### 5. Set Up Chat UI

In a new terminal:

```bash
./setup_chat_ui.sh
cd chat-ui
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

## How It Works

### Supervisor Architecture

The supervisor agent coordinates between two specialized agents:

1. **Route Analysis**: Analyzes user input to determine which agent to use
2. **Agent Delegation**: Routes weather questions to weather_agent, math to calculator_agent
3. **Response Coordination**: Combines and presents results from sub-agents

### Agent Communication

- **Weather Agent**: Handles location-based weather queries (mock data)
- **Calculator Agent**: Performs safe mathematical calculations using AST parsing
- **Supervisor**: Uses LangGraph Supervisor library for orchestration

### Chat UI Integration

- Next.js app connects to LangGraph server on port 2024
- Real-time streaming of agent conversations
- Visualizes agent handoffs and responses

## Usage Examples

**Weather Queries:**
- "What's the weather in Tokyo?"
- "How's the weather in Paris today?"

**Math Calculations:**
- "Calculate 150 + 275"
- "What's 12 * 34?"
- "Compute 100 / 4"

**General Questions:**
- "Hello, what can you help me with?"
- The supervisor will respond directly for non-specialized queries

## Architecture Benefits

- **Modularity**: Each agent handles specific tasks
- **Scalability**: Easy to add more specialized agents
- **Visibility**: Chat UI shows agent interactions and handoffs
- **Flexibility**: Supervisor can handle both delegation and direct responses